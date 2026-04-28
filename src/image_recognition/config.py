from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImageRecognitionConfig:
    image_path: Path = Path("data/images/sample.jpg")
    model_name: str = "google/vit-base-patch16-224"
    top_k: int = 3
