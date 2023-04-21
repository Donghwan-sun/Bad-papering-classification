# 데이콘 도배하자 분류 대회
한솔데코에서 주최하여 데이콘에서 진행하는 AI 경진대회

## 주제  
도배 하자의 유형을 분류는 AI 모델

## 설명  
총 19개의 도배 하자 유형을 분류하는 AI모델 개발 경진 대회

## 개발환경
 - 언어: python 3.8
 - 프레임워크: torch 2.0
 - 필요 라이브러리: efficientnet-pytorch: 0.7.1, Pillow: 9.3.0

## 설치
```
    pip install requirements.txt
    torch 에러시 https://pytorch.org/get-started/locally/ 접속하여 본인 환경에 맞게 설정후 설치
```
## 설정
```commandline
# common.config.py

class Settings:
    # 경로 설정값
    MODEL_EXTENSION: str = ".pth"
    MODEL_PATH: str = os.path.join(app_root_path, "model")
    TRAIN_DATA_PATH = os.path.join(app_root_path, r"datas\train")
    TEST_DATA_PATH = os.path.join(app_root_path, r"datas\test")
    RESULT_SAMPLE_PATH = os.path.join(app_root_path, r"datas\sample_submission.csv")
    RESULT_FILE_PATH = os.path.join(app_root_path, r"datas\result")

    # 학습 하이퍼 파라미터 설정값
    LEARNING_LATE: float = 0.003
    SPLIT_VALUE: float = 0.2
    EPOCHS: int = 30
    IMAGE_SIZE: tuple = (224, 224)
    TRAIN_RATIO: float = 0.8
    VALIDATION_RATIO: float = 0.2
    TEST_RATIO: float = None
    SHUFFLE: bool = True
    BATCH_SIZE: int = 16

    # 사전학습 모델 로드
    PRETRAINED_VERSION = 'efficientnet-b0'

    # 타겟값
    TARGET: list = ['가구수정', '걸레받이수정', '곰팡이', '꼬임', '녹오염', '들뜸', '면불량',
                    '몰딩수정', '반점', '석고수정', '오염', '오타공', '울음', '이음부불량',
                    '창틀,문틀수정', '터짐', '틈새과다', '피스', '훼손']
```

