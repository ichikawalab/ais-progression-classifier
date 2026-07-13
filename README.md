# AIS Progression Classifier

PyTorch Lightning and timm implementation for progression/non-progression
classification from frontal spine radiographs using patient-level 10-fold
cross-validation.

> **Research use only.** This software is not a validated medical device and
> must not be used for clinical decision-making.

## Evaluation protocol

Each run performs one patient-level stratified 10-fold experiment:

- 8 folds for training
- 1 fold for validation and `best.ckpt` selection
- 1 fold for test evaluation
- fixed classification threshold of `0.5`
- class weights computed from the training subset only

Every patient is used for test once and validation once. Test predictions are
generated only from the checkpoint with the lowest validation loss. The final
`predictions.csv` contains one out-of-fold prediction per sample.

### Reproducibility

Mixed-precision training (`train.precision` defaults to `bf16-mixed`) is not
bit-reproducible across GPUs; use `32-true` for strict reproducibility.
Evaluation and `ais-predict` always run in fp32.

## Data format

Training data must be provided as a CSV with four columns:

```csv
sample_id,patient_id,image_path,label
case001,patient001,C:/data/case001_Front.png,0
case002,patient002,C:/data/case002_Front.png,1
```

- `sample_id`: unique non-identifying sample ID
- `patient_id`: non-identifying patient ID used to prevent leakage
- `image_path`: absolute path or path relative to the CSV
- `label`: `0` for non-progression or `1` for progression

Multiple images from the same patient must use the same `patient_id`.

Patient images, cohort data, outputs, and trained weights are not included and
must never be committed to this repository. Only synthetic CSV examples under
`examples/` are tracked.

## Installation

Python 3.10--3.12 and [uv](https://docs.astral.sh/uv/) are recommended.

```powershell
git clone https://github.com/ichikawalab/ais-progression-classifier.git
cd ais-progression-classifier
uv sync
```

For NVIDIA GPU training, install the PyTorch build appropriate for your system
using the [official PyTorch instructions](https://pytorch.org/get-started/locally/).

Verify CUDA on Windows:

```powershell
uv run python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda, torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Optional preprocessing

The preprocessing command applies grayscale conversion, CLAHE, and square
padding to supported raster images:

```powershell
ais-preprocess --input-csv data/raw_train.csv --output-dir data/processed --output-csv data/train_processed.csv
```

Do not apply CLAHE twice. DICOM decoding, windowing, and de-identification are
not implemented; DICOM data must be converted using a validated local workflow.

## Training

```powershell
ais-train --data-csv data/train.csv --run-name resnet50_seed42
```

Common options:

```powershell
ais-train --data-csv data/train.csv --arch resnet50 --batch-size 16 --seed 42 --num-workers 4 --run-name experiment1
```

Defaults are defined in `configs/default.yaml`. Batch size may be reduced to fit
GPU memory. Resume an interrupted run with:

```powershell
ais-train --resume-run outputs/experiment1
```

## Outputs

```text
outputs/<run_name>/
|-- config.yaml
|-- environment.json
|-- folds.csv
|-- predictions.csv
|-- patient_predictions.csv
|-- metrics_by_fold.csv
|-- metrics_summary.json
`-- fold_00/ ... fold_09/
    |-- split.csv
    |-- predictions.csv
    |-- metrics.json
    `-- checkpoints/
        |-- best.ckpt
        `-- last.ckpt
```

Reported metrics include AUROC, AUPRC, sensitivity, specificity, accuracy,
balanced accuracy, precision, NPV, F1, Brier score, and patient-level bootstrap
confidence intervals.

## Prediction

Prediction on new images averages probabilities from all ten fold
`best.ckpt` models:

```powershell
ais-predict --run-dir outputs/experiment1 --input-csv data/new_cases.csv --output-csv predictions.csv --batch-size 16 --num-workers 0
```

Input images must use the same preprocessing as training, including the square
padding from `ais-preprocess` (non-square inputs are distorted on resize). Load
only checkpoints produced by a trusted local run.

## Grad-CAM

```powershell
ais-gradcam --run-dir outputs/experiment1 --fold 0 --target test
```

Supported model families are ResNet, DenseNet, Inception, ConvNeXt,
EfficientNet, Swin, and ViT. Grad-CAM is exploratory and does not establish a
causal explanation for a prediction.

## Tests

```powershell
uv sync --extra dev
uv run pytest -q
uv run ruff check .
```

GitHub Actions runs tests on Python 3.10, 3.11, and 3.12 and verifies the built
package and command-line entry points.

## Limitations

- Cross-validation is not external validation.
- Performance may not generalize across institutions, scanners, or populations.
- Model probabilities are not clinically calibrated risk estimates.
- Architecture selection based on OOF test performance requires an independent
  external test cohort or a nested validation design.

## License and citation

MIT License. See [LICENSE](LICENSE) and [CITATION.cff](CITATION.cff).
