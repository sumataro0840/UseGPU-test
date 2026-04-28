from .classifier import ImageClassifier
from .config import ImageRecognitionConfig


class ImageRecognitionApp:
    def __init__(self, config: ImageRecognitionConfig | None = None) -> None:
        self.config = config or ImageRecognitionConfig()
        self.classifier = ImageClassifier(
            model_name=self.config.model_name,
            top_k=self.config.top_k,
        )

    def run(self) -> list[dict[str, float | str]]:
        results = self.classifier.classify(self.config.image_path)
        self._display(results)
        return results

    def _display(self, results: list[dict[str, float | str]]) -> None:
        print("Image recognition result:")
        for index, result in enumerate(results, start=1):
            label = result["label"]
            score = float(result["score"])
            print(f"{index}. {label}: {score:.2%}")
