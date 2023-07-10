# 좀 잘되는거 사람 얼굴로 앞을 보는지 체크

import cv2
import numpy as np
import matplotlib.cm
from openvino.inference_engine import IECore
from openvino.runtime import Core


DEVICE = "AUTO"

# monodepth 사용하는 모델
monodepth_model_name = ".\model\MiDaS_small.xml"
ie1 = Core()
ie1.set_property({'CACHE_DIR': '../cache'})  # 두 번째 로딩부터 빠르게 로딩하기 위해 캐시 사용
mono_model = ie1.read_model(monodepth_model_name)
compiled_mono_model = ie1.compile_model(model=mono_model, device_name=DEVICE)

mono_input_key = compiled_mono_model.input(0)
mono_output_key = compiled_mono_model.output(0)

network_input_shape = list(mono_input_key.shape)
network_image_height, network_image_width = network_input_shape[2:]


# Inference Engine API 로드
ie = IECore()

# 얼굴 검출 모델 로드
face_detection_model_xml = "./model/face-detection-adas-0001.xml"
face_detection_model_bin = "./model/face-detection-adas-0001.bin"
face_detection_net = ie.read_network(model=face_detection_model_xml, weights=face_detection_model_bin)

# 헤드 포즈 추정 모델 로드
head_pose_model_xml = "./model/head-pose-estimation-adas-0001.xml"
head_pose_model_bin = "./model/head-pose-estimation-adas-0001.bin"
head_pose_net = ie.read_network(model=head_pose_model_xml, weights=head_pose_model_bin)

# 모델을 지정된 장치로 로드
face_detection_exec_net = ie.load_network(network=face_detection_net, device_name=DEVICE)
head_pose_exec_net = ie.load_network(network=head_pose_net, device_name=DEVICE)

# 각 모델의 입력과 출력 레이어 이름 가져오기
face_detection_input_blob = next(iter(face_detection_net.input_info))
face_detection_output_blob = next(iter(face_detection_net.outputs))
head_pose_input_blob = next(iter(head_pose_net.input_info))
head_pose_output_blob_yaw = "angle_y_fc"
head_pose_output_blob_pitch = "angle_p_fc"
head_pose_output_blob_roll = "angle_r_fc"


## monodepth에서 사용하는 Functions
def normalize_minmax(data):
    return (data - data.min()) / (data.max() - data.min())


def convert_result_to_image(result, colormap="viridis"):
    cmap = matplotlib.cm.get_cmap(colormap)
    result = result.squeeze(0)
    result = normalize_minmax(result)
    result = cmap(result)[:, :, :3] * 255
    result = result.astype(np.uint8)
    return result


def to_rgb(image_data) -> np.ndarray:
    return cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)


# 카메라에서 비디오 캡처
cap = cv2.VideoCapture(0)

w = 640
h = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)


# 저장할 동영상 파일 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 코덱 설정
output_filename = 'output.mp4'  # 저장할 동영상 파일 이름
fps = 24  # 저장할 동영상의 프레임 속도 설정
output = cv2.VideoWriter(output_filename, fourcc, fps, (w, h))

while True:
    # 비디오 캡처에서 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break

    # 얼굴 검출 모델을 위해 프레임 전처리
    face_detection_input_frame = cv2.resize(frame, (672, 384))  # 크기 조정 (W, H)
    face_detection_input_frame = face_detection_input_frame.transpose((2, 0, 1))  # 채널 우선 순서 (C, H, W)
    face_detection_input_frame = np.expand_dims(face_detection_input_frame, axis=0)  # 배치 차원 추가

    # 얼굴 검출 추론 실행
    face_detection_results = face_detection_exec_net.infer(inputs={face_detection_input_blob: face_detection_input_frame})
    face_detection_output = face_detection_results[face_detection_output_blob]

    # 검출된 얼굴의 좌표와 신뢰도 가져오기
    face_coordinates = face_detection_output[0, 0, :, 3:7]
    face_confidences = face_detection_output[0, 0, :, 2]

    for i, face_confidence in enumerate(face_confidences):
        if face_confidence > 0.5:  # 신뢰도 임계값 조정
            x_min, y_min, x_max, y_max = face_coordinates[i] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
            # 프레임에서 얼굴 영역 추출
            face_region = frame[y_min:y_max, x_min:x_max]
            if not face_region.size == 0:  # 얼굴 영역이 비어있지 않은지 확인
                head_pose_input_frame = cv2.resize(face_region, (60, 60))
                head_pose_input_frame = head_pose_input_frame.transpose((2, 0, 1))  # 채널 우선 순서 (C, H, W)
                head_pose_input_frame = np.expand_dims(head_pose_input_frame, axis=0)  # 배치 차원 추가

                # 헤드 포즈 추정 추론 실행
                head_pose_results = head_pose_exec_net.infer(inputs={head_pose_input_blob: head_pose_input_frame})
                yaw = head_pose_results[head_pose_output_blob_yaw][0][0]  # 얼굴 좌우 각도
                pitch = head_pose_results[head_pose_output_blob_pitch][0][0]  # 얼굴 상하 각도
                roll = head_pose_results[head_pose_output_blob_roll][0][0]  # 얼굴 기울기 각도

                # 사람이 카메라를 바라보고 있는지 여부 확인
                threshold = 30  # 필요에 따라 임계값 조정
                # 뒤에 주석처리 한 이유는 정확도 떄문에 그냥 좌우만 판단한다.
                facing_camera = abs(yaw) < threshold #and abs(pitch) < threshold and abs(roll) < threshold


                # monodepth 사용
                resized_image = cv2.resize(src=frame, dsize=(network_image_height, network_image_width))
                input_image = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0)
                result = compiled_mono_model([input_image])[mono_output_key]
                result_frame = to_rgb(convert_result_to_image(result))
                depth_map = result_frame.squeeze()
                depth_map_m = depth_map.mean()

            # 얼굴 영역 사각형
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)

            # 가까이 있는 경우
            if depth_map_m > 90:
                # 사람이 카메라쪽을 보는 경우
                if facing_camera:         
                    cv2.putText(img=frame, text='Die XoX', org=(0,45), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale=2, color = (0,0,255), thickness=2)
                else:
                    cv2.putText(img=frame, text='Dangerous', org=(0,45), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                            fontScale=2, color = (0,128,255), thickness=2)
            else:
                if facing_camera:         
                    cv2.putText(img=frame, text='Dangerous', org=(0,45), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale=2, color = (153,51,255), thickness=2)
                else:
                    cv2.putText(img=frame, text='Safe', org=(0,45), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale=2, color = (0,255,0), thickness=2)



    # 원래 사이즈로 변경
    frame = cv2.resize(frame, (640,480))

    # 동영상 파일에 프레임 쓰기
    output.write(frame)

    cv2.imshow("Main", frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# 비디오 캡처 해제 및 창 닫기
cap.release()
output.release()
cv2.destroyAllWindows()