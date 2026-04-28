from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ObjectDetectionConfig:
    video_path: Path = Path("data/videos/sample.mp4")
    output_dir: Path = Path("data/results/videos")
    model_name: str = "models/shuttlecock.pt"
    confidence: float = 0.25
    image_size: int = 1280

    @property
    def output_path(self) -> Path:
        return self.output_dir / f"{self.video_path.stem}_detected.mp4"
