from torchvision.transforms import RandomHorizontalFlip, RandomResizedCrop

from ais_progression.config import AugmentConfig
from ais_progression.data import build_transforms


def test_default_training_augmentation():
    transform = build_transforms(
        {"input_size": (3, 224, 224), "mean": (0.1, 0.2, 0.3), "std": (1, 1, 1)},
        AugmentConfig(),
        is_training=True,
    )
    assert isinstance(transform.transforms[0], RandomHorizontalFlip)
    assert transform.transforms[0].p == 0.5
    assert isinstance(transform.transforms[1], RandomResizedCrop)
    assert transform.transforms[1].scale == (0.5, 1.0)
    assert transform.transforms[1].ratio == (1.0, 1.0)
