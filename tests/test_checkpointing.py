import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, TensorDataset

from ais_progression.checkpointing import load_trusted_checkpoint
from ais_progression.config import ModelConfig, TrainConfig
from ais_progression.lit_module import TransferLightningModule


def test_checkpoint_roundtrip_preserves_logits(tmp_path):
    torch.manual_seed(0)
    images = torch.rand(4, 3, 32, 32)
    labels = torch.tensor([0, 1, 0, 1])
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)
    module = TransferLightningModule(
        ModelConfig(arch="resnet18", pretrained=False, hidden_dim=8, dropout=0.0),
        TrainConfig(max_epochs=1, min_epochs=0, warmup_epochs=0),
        class_weights=[1.0, 1.0],
    )
    trainer = Trainer(
        max_epochs=1,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(module, train_dataloaders=loader)
    module.eval()
    with torch.inference_mode():
        expected = module(images[:2])

    checkpoint = tmp_path / "model.ckpt"
    trainer.save_checkpoint(checkpoint)
    restored = load_trusted_checkpoint(checkpoint, torch.device("cpu")).eval()
    with torch.inference_mode():
        actual = restored(images[:2])
    torch.testing.assert_close(actual, expected)
