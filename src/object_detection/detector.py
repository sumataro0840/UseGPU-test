from pathlib import Path


class ObjectDetector:
    def __init__(self, model_name: str = "models/shuttlecock.pt", confidence: float = 0.25) -> None:
        self.model_name = model_name
        self.confidence = confidence
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics が入っていません。`pip install -r requirements.txt` を実行してください。"
            ) from exc

        model_path = Path(self.model_name)
        if model_path.suffix == ".pt" and not model_path.exists() and self.model_name != "yolov8n.pt":
            raise RuntimeError(
                f"{self.model_name} が見つかりません。先に `python3 main.py --mode train-shuttle` で学習するか、"
                "`--model yolov8n.pt` を指定して一般物体検出を試してください。"
            )

        self._model = YOLO(self.model_name)
        return self._model

    def detect(self, frame):
        model = self._load_model()
        results = model(frame, conf=self.confidence, verbose=False)
        return results[0]
