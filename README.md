# 너 지금 뭐하니?

* 교수님의 접근을 파악하여 수업중 딴짓을 완벽하게 하기

## Requirement

```
* 9th generation Intel® CoreTM processor onwards
* At least 32GB RAM
* Windows 10
* Python 3.9
```

## Clone code

```shell
git clone https://github.com/MoonByungBok/Intel.AI.TeamProject.git
```

## Prerequite

```shell
python -m venv openvino_env
openvino_env\Scripts\activate

git clone --depth=1 https://github.com/openvinotoolkit/openvino_notebooks.git
cd openvino_notebooks

python -m pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
```

## Steps to build

```shell
cd works
openvino_env\Scripts\activate
```

## Steps to run

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
