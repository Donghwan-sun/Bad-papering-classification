import os
import pathlib

# 프로젝트 루트 path
app_root_path = pathlib.Path(os.path.abspath(__file__)).parent.parent

class Settings:
    """
        스크립트 설정값 모음
    """
    # 경로 설정값
    MODEL_EXTENSION: str = ".pth"
    MODEL_PATH: str = os.path.join(app_root_path, "models")
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


settings = Settings()
