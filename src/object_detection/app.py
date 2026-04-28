from .config import ObjectDetectionConfig
from .detector import ObjectDetector
from .video_processor import VideoProcessor


class ObjectDetectionApp:
    def __init__(self, config: ObjectDetectionConfig | None = None) -> None:
        self.config = config or ObjectDetectionConfig()
        detector = ObjectDetector(
            model_name=self.config.model_name,
            confidence=self.config.confidence,
            image_size=self.config.image_size,
        )
        self.video_processor = VideoProcessor(detector)

    def run(self):
        output_path = self.video_processor.process(
            input_path=self.config.video_path,
            output_path=self.config.output_path,
        )
        print("Object detection result:")
        print(f"Saved: {output_path}")
        return output_path
