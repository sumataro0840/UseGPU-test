from pathlib import Path


class ImageClassifier:
    def __init__(self, model_name: str, top_k: int = 3) -> None:
        self.model_name = model_name
        self.top_k = top_k
        self._pipeline = None

    def _load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline

        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "transformers が入っていません。`pip install -r requirements.txt` を実行してください。"
            ) from exc

        self._pipeline = pipeline("image-classification", model=self.model_name)
        return self._pipeline

    def classify(self, image_path: Path) -> list[dict[str, float | str]]:
        if not image_path.exists():
            raise FileNotFoundError(
                f"{image_path} が見つかりません。画像を data/images/sample.jpg に置いてください。"
            )

        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "Pillow が入っていません。`pip install -r requirements.txt` を実行してください。"
            ) from exc

        image = Image.open(image_path)
        classifier = self._load_pipeline()
        return classifier(image, top_k=self.top_k)
