from pathlib import Path

from .detector import ObjectDetector


class VideoProcessor:
    def __init__(self, detector: ObjectDetector) -> None:
        self.detector = detector

    def process(self, input_path: Path, output_path: Path) -> Path:
        if not input_path.exists():
            raise FileNotFoundError(
                f"{input_path} が見つかりません。動画を data/videos/sample.mp4 に置いてください。"
            )

        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "opencv-python が入っていません。`pip install -r requirements.txt` を実行してください。"
            ) from exc

        output_path.parent.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(str(input_path))

        if not capture.isOpened():
            raise RuntimeError(f"{input_path} を動画として開けませんでした。")

        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        frame_count = 0
        detected_frame_count = 0
        detected_box_count = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                result = self.detector.detect(frame)
                box_count = len(result.boxes)
                if box_count > 0:
                    detected_frame_count += 1
                    detected_box_count += box_count
                annotated_frame = result.plot()
                writer.write(annotated_frame)
                frame_count += 1
        finally:
            capture.release()
            writer.release()

        if frame_count == 0:
            raise RuntimeError(f"{input_path} からフレームを読み込めませんでした。")

        print(f"Processed frames: {frame_count}")
        print(f"Detected frames: {detected_frame_count}")
        print(f"Detected boxes: {detected_box_count}")
        return output_path
