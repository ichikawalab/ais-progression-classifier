from pathlib import Path

import pandas as pd
import torch
from PIL import Image

import ais_progression.commands.train as train_command


class FakeCheckpoint:
    def __init__(self, dirpath, **_):
        self.dirpath = Path(dirpath)
        self.best_model_path = ""
        self.best_model_score = None


class FakeDataModule:
    def __init__(self, *, test_df, **_):
        self.test_df = test_df

    def test_dataloader(self):
        return None


class FakeTrainer:
    def __init__(self, *, callbacks, **_):
        self.checkpoint = next(item for item in callbacks if isinstance(item, FakeCheckpoint))

    def fit(self, *_args, **_kwargs):
        self.checkpoint.dirpath.mkdir(parents=True, exist_ok=True)
        best = self.checkpoint.dirpath / "best.ckpt"
        last = self.checkpoint.dirpath / "last.ckpt"
        best.write_bytes(b"synthetic checkpoint")
        last.write_bytes(b"synthetic checkpoint")
        self.checkpoint.best_model_path = str(best)
        self.checkpoint.best_model_score = torch.tensor(0.1)


def fake_predictions(_model, _loader, source_df, _device, fold, threshold=0.5):
    probability = source_df["label"].astype(float).to_numpy() * 0.8 + 0.1
    return pd.DataFrame(
        {
            "sample_id": source_df["sample_id"].to_numpy(),
            "patient_id": source_df["patient_id"].to_numpy(),
            "true_label": source_df["label"].to_numpy(),
            "fold": fold,
            "prob_class0": 1 - probability,
            "prob_class1": probability,
            "predicted_label": (probability >= threshold).astype(int),
        }
    )


def test_train_command_produces_complete_ten_fold_oof_artifacts(tmp_path, monkeypatch):
    rows = []
    for index in range(20):
        image_path = tmp_path / f"image_{index:02d}.png"
        Image.new("L", (8, 8), color=index).save(image_path)
        rows.append(
            {
                "sample_id": f"sample-{index:02d}",
                "patient_id": f"patient-{index:02d}",
                "image_path": image_path.name,
                "label": index % 2,
            }
        )
    csv_path = tmp_path / "train.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    monkeypatch.setattr(train_command, "AISDataModule", FakeDataModule)
    monkeypatch.setattr(train_command, "TransferLightningModule", lambda *_args: object())
    monkeypatch.setattr(train_command, "ModelCheckpoint", FakeCheckpoint)
    monkeypatch.setattr(train_command, "EarlyStopping", lambda **_: object())
    monkeypatch.setattr(train_command, "LearningRateMonitor", lambda **_: object())
    monkeypatch.setattr(train_command, "TensorBoardLogger", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(train_command.pl, "Trainer", FakeTrainer)
    monkeypatch.setattr(train_command, "load_trusted_checkpoint", lambda *_args: object())
    monkeypatch.setattr(train_command, "predict_dataframe", fake_predictions)
    monkeypatch.setattr(
        train_command,
        "bootstrap_confidence_intervals",
        lambda *_args, **_kwargs: {"auroc": [1.0, 1.0]},
    )

    output_dir = tmp_path / "outputs"
    train_command.main(
        [
            "--data-csv",
            str(csv_path),
            "--output-dir",
            str(output_dir),
            "--run-name",
            "integration",
            "--num-workers",
            "0",
        ]
    )

    run_dir = output_dir / "integration"
    predictions = pd.read_csv(run_dir / "predictions.csv")
    assert len(predictions) == 20
    assert predictions["sample_id"].is_unique
    assert set(predictions["fold"]) == set(range(10))
    for fold in range(10):
        fold_dir = run_dir / f"fold_{fold:02d}"
        assert (fold_dir / "checkpoints" / "best.ckpt").exists()
        assert (fold_dir / "predictions.csv").exists()
