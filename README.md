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

The `full` profile reads a patient-level CSV with these columns:

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

Profiles that do not use an image modality may omit its path column.

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

For comparison, `--method` also accepts `average`, `logreg`, `svm`, and `rf`.

Re-running the same command resumes incomplete folds. Use `--no-resume` to
recompute them.

### Outputs

Cross-validation results are written under `outputs/cv/` and ensemble results
under `outputs/ensemble/`. Each run includes predictions, fold metrics, and
`summary.json`. The headline result is `test_auc_pooled_per_rep`, reported as
the mean and SD across repetitions.

## Final model

```cmd
ais-train-final --bundle-dir outputs\final
```

This trains the required models on the entire input cohort. Image models use the
median epoch count from cross-validation.

Prediction requires a trained bundle. Model weights are not included in this
repository; create a bundle with the command above or use one supplied by a
trusted source.

## Prediction

The bundle provides three input profiles:

| Profile | Required input | Models |
| --- | --- | --- |
| `full` (default) | Frontal image, lateral image, clinical variables | All 9 models |
| `front_clinical` | Frontal image, clinical variables | 6 models |
| `clinical_only` | Clinical variables | 3 models |

List the profiles stored in a bundle:

```cmd
ais-predict --bundle-dir outputs\final --list-profiles
```

For `full`, first preprocess the new radiographs:

```cmd
ais-preprocess --dataset-csv data\new_cases.csv --output-dir data\new_cases_processed --output-csv data\new_cases_processed.csv
```

For `front_clinical`, add `--modalities front`; `clinical_only` requires no
image preprocessing.

Then run the default `full` profile:

```cmd
ais-predict --bundle-dir outputs\final --input-csv data\new_cases_processed.csv --output-csv outputs\predictions.csv
```

Clinical-only prediction does not require images or preprocessing:

```cmd
ais-predict --bundle-dir outputs\final --profile clinical_only --input-csv data\new_clinical_cases.csv --output-csv outputs\clinical_predictions.csv
```

`patient_id` is always required and `label` is optional. When labels are
present, AUC is reported. Missing or out-of-range clinical values are rejected.
Do not preprocess an image more than once because CLAHE is not idempotent.

The main output columns are:

| Column | Meaning |
| --- | --- |
| `probability` | Raw weighted-ensemble score used for classification |
| `calibrated_probability` | Internally calibrated reference probability |
| `predicted_label` | Binary prediction from `probability` and the profile threshold |
| `threshold` | Threshold used for `predicted_label` |
| `profile` | Profile used for prediction |

`calibrated_probability` has not been externally validated as a clinical risk
estimate. Loading a bundle unpickles scikit-learn pipelines, so only use a
bundle from a trusted source.

## Grad-CAM

```cmd
ais-gradcam --bundle-dir outputs\final --modality front --model convnextv2 --input-csv data\dataset.csv --limit 20
```

Grad-CAM supports the configured ViT, Swin, and ConvNeXtV2 backbones. It targets
progression class (`--target-class 1`) by default; use `--target-class pred` to
visualise each model's predicted class. Grad-CAM is exploratory and does not
establish a causal explanation for a prediction.

## Configuration

Defaults live in [configs/default.yaml](configs/default.yaml). Override settings
from the command line with `--set`:

```cmd
ais-cv-modality --modality front --model vit --set train.max_epochs=50 --set data.batch_size=8 --reps 2
```

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
