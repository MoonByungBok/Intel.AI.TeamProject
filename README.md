# 너 지금 뭐하니?

* 교수님의 접근을 파악하여 수업중 딴짓을 완벽하게 하기
* (간략히 전체 프로젝트를 설명하고, 최종 목표가 무엇인지에 대해 기술)

## Requirement

* (프로젝트를 실행시키기 위한 최소 requirement들에 대해 기술)

```
* 9th generation Intel® CoreTM processor onwards
* At least 32GB RAM
* Windows 10
* Python 3.9
```

## Clone code

* (Code clone 방법에 대해서 기술)

```shell
git clone https://github.com/MoonByungBok/Intel.AI.TeamProject
```

## Prerequite

* (프로잭트를 실행하기 위해 필요한 dependencies 및 configuration들이 있다면, 설치 및 설정방법에 대해 기술)

```shell
python -m venv openvino_env
openvino_env\Scripts\activate

git clone --depth=1 https://github.com/openvinotoolkit/openvino_notebooks.git
cd openvino_notebooks

python -m pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
```

## Steps to build

* (프로젝트를 실행을 위해 빌드 절차 기술)

```shell
cd works
openvino_env\Scripts\activate
```

## Steps to run

* (프로젝트 실행방법에 대해서 기술, 특별한 사용방법이 있다면 같이 기술)

```shell
cd works
openvino_env\Scripts\activate

cd .\Project
python project_final.py
```

## Output

![./images/result.jpg](./images/result.jpg)

## Appendix

* (참고 자료 및 알아두어야할 사항들 기술)
