import os
import pathlib

# 프로젝트 루트 path
app_root_path = pathlib.Path(os.path.abspath(__file__)).parent.parent

class Settings:
    """
        스크립트 설정값 모음
    """
    # model 파일 설정값
    MODEL_EXTENSION = ".pth"
    MODEL_PATH: str = os.path.join(app_root_path, "models/")
    

settings = Settings()
