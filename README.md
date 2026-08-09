# AIS Progression Classifier

Multimodal ensemble prediction of curve progression in idiopathic scoliosis from
frontal and lateral whole-spine radiographs plus clinical variables.

> **Research use only.** This software is not a validated medical device and
> must not be used for clinical decision-making.

This repository contains code related to:

Arima H, Ichikawa S, et al. *Development and Validation of a Multi-Modal
Ensemble Model for Predicting Progression in Idiopathic Scoliosis*. Global Spine
Journal. Published online July 31, 2026.
[https://doi.org/10.1177/21925682261474876](https://doi.org/10.1177/21925682261474876)

## What the pipeline does

Nine individual models across three modalities, combined by late fusion:

| Modality | Models |
| --- | --- |
| Frontal radiograph | ViT, Swin Transformer, ConvNeXtV2 |
| Lateral radiograph | ViT, Swin Transformer, ConvNeXtV2 |
| Clinical variables | Logistic regression, SVM, random forest |

Their predicted probabilities are combined by weighted averaging. Simple
averaging, logistic regression, SVM, and random forest are also available for
comparison.

There are two paths through the code:

* **Cross-validation** estimates performance but does not produce a deployable
  model.
* **The final model** is trained on the whole cohort and packaged as a bundle
  for inference. Its epoch counts, ensemble weights, thresholds, and calibrators
  are derived from cross-validation.

### Why the final model has no holdout

The final model uses the entire input cohort. It has no independent performance
estimate of its own; reported performance comes from cross-validation.

## Package layout

```text
src/ais_progression/
|-- config.py         typed configuration; defaults are the reference settings
|-- evaluation.py     AUC and threshold-dependent metrics
|-- utils.py          seeding, device selection, run metadata
|-- data/             dataset schema, preprocessing, and loaders
|-- models/           image classifiers (timm + Lightning) and clinical models
|-- ensemble/         weighted averaging and the stacked comparators
|-- experiments/      splitting, repeated nested cross-validation, reporting
|-- final/            full-cohort training, the model bundle, inference
`-- cli/              one module per command
```

## Installation

Python 3.11-3.12 and [uv](https://docs.astral.sh/uv/).

```cmd
git clone https://github.com/ichikawalab/ais-progression-classifier.git
cd ais-progression-classifier
uv sync
call .venv\Scripts\activate.bat
```

The commands below assume Windows `cmd.exe` with this environment active.

### GPU training

On Windows, replace the PyPI CPU wheels with the appropriate CUDA build after
`uv sync`:

```cmd
uv pip install --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu130
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

Use the CUDA index recommended by the
[official PyTorch instructions](https://pytorch.org/get-started/locally/); `cu130`
matches the currently locked Windows build. Do not run `uv sync` or plain
`uv run` afterwards because either may restore the locked CPU wheels. The
commands below call the active environment directly.

## Data format

Everything downstream reads a single patient-level CSV:

```csv
patient_id,front_path,lateral_path,age,sex,risser,cobb_baseline,label
case001,front/case001.png,lateral/case001.png,11.4,2,0,31.0,1
case002,front/case002.png,lateral/case002.png,13.8,2,4,22.5,0
```

* `patient_id` - non-identifying ID, unique within the file
* `front_path`, `lateral_path` - relative to the CSV (recommended) or absolute
* `age` - years at the initial visit
* `sex` - 1 male, 2 female
* `risser` - Risser sign 0-5, treated as ordinal
* `cobb_baseline` - baseline Cobb angle in degrees
* `label` - 0 non-progression (<=5 deg), 1 progression (>=10 deg)

Borderline patients (6-9 deg) are excluded before this file is built. See
[examples/sample_dataset.csv](examples/sample_dataset.csv).

Patient images, cohort data, outputs, and trained weights are never committed.
Only the synthetic CSVs under `examples/` are tracked.

## Preprocessing

CLAHE followed by zero-padding to a square canvas. Intensity normalisation with
the ImageNet mean and standard deviation happens later, in the training
transform.

```cmd
ais-preprocess --dataset-csv data\dataset.csv --output-dir data\processed --output-csv data\dataset_processed.csv
```

Do not run this twice on the same cohort: CLAHE is not idempotent. DICOM
decoding, windowing, and de-identification are not implemented; convert DICOM
data with a validated local workflow first.

## Cross-validation

The default evaluation is repeated stratified 10-fold cross-validation with 10
repetitions. Image models use a validation slice for early stopping; clinical
and ensemble models use inner 10-fold Optuna tuning. Seeds and split metadata
are recorded for every fold, and interrupted runs can be resumed.

Run each of the nine individual models:

```cmd
ais-cv-modality --modality front --model vit
ais-cv-modality --modality front --model swint
ais-cv-modality --modality front --model convnextv2
ais-cv-modality --modality lateral --model vit
ais-cv-modality --modality lateral --model swint
ais-cv-modality --modality lateral --model convnextv2
ais-cv-modality --modality clinical --model logreg
ais-cv-modality --modality clinical --model svm
ais-cv-modality --modality clinical --model rf
```

Then the ensembles, which read every completed run under `outputs/cv/`:

```cmd
ais-cv-ensemble --method weighted
```

Additional comparison methods are available:

```cmd
ais-cv-ensemble --method average
ais-cv-ensemble --method logreg
ais-cv-ensemble --method svm
ais-cv-ensemble --method rf
```

Re-running the same command resumes incomplete folds. Use `--no-resume` to
recompute or `--keep-checkpoints` to retain fold weights.

To restrict the ensemble to a subset of modalities, name the base models
explicitly:

```cmd
ais-cv-ensemble --method weighted --base front_vit=outputs\cv\front_vit --base clinical_logreg=outputs\cv\clinical_logreg --run-dir outputs\ensemble\front_clinical
```

### Outputs

```text
outputs/
|-- cv/<modality>_<model>/
|   |-- config.yaml
|   |-- environment.json
|   |-- folds/rep01_fold01.csv, rep01_fold01.json, ...
|   |-- predictions.csv      patient_id, rep, fold, split, true_label, prob
|   |-- fold_metrics.csv     per fold: selection_auc, test_auc, best_epoch, sizes
|   `-- summary.json
`-- ensemble/<method>/
    |-- predictions.csv
    |-- fold_metrics.csv
    |-- weights_by_fold.csv     (weighted only)
    |-- weights_summary.json    (weighted only) per model and per modality
    `-- summary.json
```

`summary.json` reports:

* `test_auc_pooled_per_rep` - the headline mean and SD across repetitions.
* `test_auc_per_fold` - test AUC of each individual fold.
* `selection_auc_by_source` - the selection AUC grouped by source (`holdout`,
  `inner_cv`, or `train`). These sources are not directly comparable.

Ensemble runs also carry an `ensemble_method_selection_warning`: comparing
several ensemble methods on the same test folds and keeping the best one adds a
separate selection bias, so the winner's AUC should be read as a selected-best
value. The stacking leakage itself is recorded separately.

## Final model

```cmd
ais-train-final --bundle-dir outputs\final
```

This trains the required models on the entire input cohort. Image models use the
median epoch count from cross-validation.

Before training, the command verifies that the selected CV runs match the
current cohort, configuration, software, source tree, and image data.

Image models are stored as bare `state_dict` tensors, which keeps the bundle
small and lets them load with `torch.load(weights_only=True)`. Pass
`--save-full-checkpoints` for full Lightning checkpoints.

### Serving profiles

A bundle can carry several model combinations. Each profile has its own weights,
threshold, calibrator, and cross-validated AUC derived from out-of-fold
predictions. Adding profiles does not repeat image training.

The defaults are `full` (every model), `front_clinical`, and `clinical_only`.
Declare your own with `--profile NAME=MODALITIES`:

```cmd
ais-train-final --profile full= --profile cheap=clinical --default-profile full
```

Use `--cv-seed` to identify CV runs made with a non-default split seed, and
`--final-seed` to change the independent seed from which final bundle members
derive their model-specific seeds.

The default decision threshold uses the median Youden threshold across
repetitions. Use `--threshold-policy target_sensitivity --target-sensitivity
0.9` to target sensitivity. Isotonic calibration is the default; alternatives
are `--calibration platt|none`.

```text
outputs/final/
|-- manifest.json        format version, model inventory, serving profiles
|-- config.yaml
|-- metrics.json         per-profile cross-validated AUC and operating point
|-- environment.json     package versions and the dataset SHA256
`-- models/
    |-- front_vit.pt ... lateral_convnextv2.pt
    |-- clinical_logreg.joblib ...
    `-- calibrator_full.joblib ...
```

## Prediction

```cmd
ais-predict --bundle-dir outputs\final --input-csv data\new_cases.csv --output-csv predictions.csv
ais-predict --bundle-dir outputs\final --list-profiles
ais-predict --bundle-dir outputs\final --profile clinical_only --input-csv data\new_cases.csv --output-csv predictions.csv
```

The input CSV needs only the fields used by the selected profile: for example,
`clinical_only` needs no image paths, while a front-only profile needs neither
the lateral path nor clinical variables. `patient_id` is always required and
`label` is optional; when present, AUC is reported. Output holds one column per
base model plus `probability`, `calibrated_probability`,
`predicted_label`, `threshold`, `profile`, and `imputed_fields`.

Missing clinical variables are **rejected by default**: imputing them silently
would return a confident-looking probability built partly from training medians.
Pass `--allow-missing` to impute anyway, and the affected fields are named in
`imputed_fields`. Values outside plausible ranges (see `FEATURE_BOUNDS` in
`data/schema.py`) are always rejected.

New images must be preprocessed the same way as training, including the square
padding from `ais-preprocess` -- non-square inputs are distorted on resize.
Loading a bundle unpickles scikit-learn pipelines, so only load bundles from a
run you trust.

### Serving from an application

`ModelBundle` caches models after their first use, so an application pays the
load cost once rather than per request:

```python
from ais_progression.final import ModelBundle

bundle = ModelBundle.load("outputs/final")
bundle.warmup("full")            # load the weights at startup
result = bundle.predict(frame)   # subsequent calls reuse them
bundle.release()                 # drop them and free GPU memory
```

## Grad-CAM

```cmd
ais-gradcam --bundle-dir outputs\final --modality front --model convnextv2 --input-csv data\dataset.csv --limit 20
```

Grad-CAM supports the configured ViT, Swin, and ConvNeXtV2 backbones. It is
exploratory and does not establish a causal explanation for a prediction.

## Configuration

Defaults live in [configs/default.yaml](configs/default.yaml) and reproduce the
reference settings: AdamW at lr 1e-5 with weight decay 1e-3, batch size 32, up
to 100 epochs with a 5-epoch linear warmup then cosine annealing, early stopping
after 5 epochs without validation improvement, inverse-frequency class weights,
384x384 inputs, and a shared head of LayerNorm, Linear(512), GELU, Dropout(0.5),
Linear(2). Augmentation is horizontal flipping (p=0.5) and a random resized crop
covering 50-100% of the image at a fixed 1:1 aspect ratio, applied to training
folds only.

Override anything from the command line:

```cmd
ais-cv-modality --modality front --model vit --set train.max_epochs=50 --set data.batch_size=8 --reps 2
```

Mixed-precision training is not bit-reproducible across GPUs. Set
`--set train.precision=32-true` for strict reproducibility; inference always
runs in fp32.

## Tests

```cmd
uv pip install pytest ruff
pytest -q
ruff check .
```

The integration tests run the whole protocol end to end on a synthetic cohort
with a small CNN, so they finish in seconds on CPU.

## Limitations

- Cross-validation is not external validation, and performance may not
  generalize across institutions, scanners, or populations.
- Comparing ensemble methods on the same test folds introduces selection bias.
  Fusion weights are fitted on one out-of-fold probability matrix, as in the
  reference procedure, so ensemble AUC may be optimistic.
- Final serving weights are refitted on all out-of-fold predictions; reported
  metrics estimate the procedure, not the exact final parameter vector.
- Calibration is fitted on this cohort and may not generalize.
- Horizontal flipping alters laterality, which may matter for right- and
  left-sided curves; curve direction is not modelled.
- Patients with a 6-9 degree increase were excluded, so the model is untested on
  borderline cases.
- Brace treatment is not a model input, and was not randomly assigned.

## License and citation

MIT License. See [LICENSE](LICENSE) and [CITATION.cff](CITATION.cff).
