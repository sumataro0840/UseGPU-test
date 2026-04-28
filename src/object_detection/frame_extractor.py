from pathlib import Path


class FrameExtractor:
    def __init__(self, frame_step: int = 15) -> None:
        self.frame_step = frame_step

    def extract(self, video_path: Path, output_dir: Path) -> int:
        if not video_path.exists():
            raise FileNotFoundError(f"{video_path} が見つかりません。")

        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "opencv-python が入っていません。`pip install -r requirements.txt` を実行してください。"
            ) from exc

        output_dir.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(str(video_path))

        if not capture.isOpened():
            raise RuntimeError(f"{video_path} を動画として開けませんでした。")

        saved_count = 0
        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                if frame_index % self.frame_step == 0:
                    output_path = output_dir / f"{video_path.stem}_{frame_index:06d}.jpg"
                    cv2.imwrite(str(output_path), frame)
                    saved_count += 1

                frame_index += 1
        finally:
            capture.release()

        if saved_count == 0:
            raise RuntimeError(f"{video_path} からフレームを抽出できませんでした。")

        return saved_count
