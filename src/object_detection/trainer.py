from pathlib import Path


class ShuttlecockTrainer:
    def __init__(
        self,
        dataset_yaml: Path = Path("data/datasets/shuttlecock/data.yaml"),
        base_model: str = "yolov8n.pt",
        output_model: Path = Path("models/shuttlecock.pt"),
        epochs: int = 50,
        image_size: int = 960,
    ) -> None:
        self.dataset_yaml = dataset_yaml
        self.base_model = base_model
        self.output_model = output_model
        self.epochs = epochs
        self.image_size = image_size

    def train(self) -> Path:
        if not self.dataset_yaml.exists():
            raise FileNotFoundError(
                f"{self.dataset_yaml} が見つかりません。先に data/datasets/shuttlecock に画像とラベルを用意してください。"
            )

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics が入っていません。`pip install -r requirements.txt` を実行してください。"
            ) from exc

        model = YOLO(self.base_model)
        result = model.train(
            data=str(self.dataset_yaml),
            epochs=self.epochs,
            imgsz=self.image_size,
            project="runs/shuttlecock",
            name="train",
        )

        best_model = Path(result.save_dir) / "weights" / "best.pt"
        if not best_model.exists():
            raise RuntimeError("学習は終わりましたが best.pt が見つかりませんでした。")

        self.output_model.parent.mkdir(parents=True, exist_ok=True)
        self.output_model.write_bytes(best_model.read_bytes())
        return self.output_model
