import argparse
from pathlib import Path

from src.image_recognition import ImageRecognitionApp, ImageRecognitionConfig
from src.object_detection import (
    FrameExtractor,
    ObjectDetectionApp,
    ObjectDetectionConfig,
    ShuttlecockTrainer,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run image recognition or video object detection.")
    parser.add_argument(
        "--mode",
        choices=("detect", "extract-frames", "train-shuttle"),
        default="detect",
        help="detect: run recognition, extract-frames: save video frames, train-shuttle: train shuttlecock detector.",
    )
    parser.add_argument(
        "--model",
        default="models/shuttlecock.pt",
        help="YOLO model path. Use models/shuttlecock.pt after training, or yolov8n.pt for general detection.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Detection confidence threshold.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=15,
        help="Save one frame every N frames when using extract-frames.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Training epochs when using train-shuttle.",
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default="data/images/sample.jpg",
        help="Path to the image or video file.",
    )
    return parser.parse_args()


def run_image_recognition(input_path: Path) -> None:
    config = ImageRecognitionConfig(image_path=input_path)
    app = ImageRecognitionApp(config)
    app.run()


def run_object_detection(input_path: Path, model_name: str, confidence: float) -> None:
    config = ObjectDetectionConfig(
        video_path=input_path,
        model_name=model_name,
        confidence=confidence,
    )
    app = ObjectDetectionApp(config)
    app.run()


def extract_frames(input_path: Path, frame_step: int) -> None:
    output_dir = Path("data/datasets/shuttlecock/images/unlabeled") / input_path.stem
    extractor = FrameExtractor(frame_step=frame_step)
    saved_count = extractor.extract(video_path=input_path, output_dir=output_dir)
    print("Frame extraction result:")
    print(f"Saved {saved_count} frames to {output_dir}")


def train_shuttlecock(epochs: int) -> None:
    trainer = ShuttlecockTrainer(epochs=epochs)
    output_model = trainer.train()
    print("Shuttlecock training result:")
    print(f"Saved: {output_model}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    extension = input_path.suffix.lower()

    try:
        if args.mode == "extract-frames":
            extract_frames(input_path, args.frame_step)
        elif args.mode == "train-shuttle":
            train_shuttlecock(args.epochs)
        elif extension in IMAGE_EXTENSIONS:
            run_image_recognition(input_path)
        elif extension in VIDEO_EXTENSIONS:
            run_object_detection(input_path, args.model, args.confidence)
        else:
            print(f"{extension or '拡張子なし'} は未対応です。画像か動画ファイルを指定してください。")
    except (FileNotFoundError, RuntimeError) as exc:
        print(exc)


if __name__ == "__main__":
    main()
