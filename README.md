# AIS Progression Classifier

Multimodal ensemble prediction of curve progression in idiopathic scoliosis from
frontal and lateral whole-spine radiographs plus clinical variables.

Reference implementation for Arima et al., *Development and Validation of a
Multi-Modal Ensemble Model for Predicting Progression in Idiopathic Scoliosis*,
Global Spine Journal (2026).

> **Research use only.** This software is not a validated medical device and
> must not be used for clinical decision-making.

## What the pipeline does

Nine individual models across three modalities, combined by late fusion:

| Modality | Models |
| --- | --- |
| Frontal radiograph | ViT, Swin Transformer, ConvNeXtV2 |
| Lateral radiograph | ViT, Swin Transformer, ConvNeXtV2 |
| Clinical variables | Logistic regression, SVM, random forest |

Their predicted probabilities are combined by weighted averaging (the
best-performing method in the paper), or by simple averaging, logistic
regression, SVM, or random forest for comparison.

There are two separate paths through the code:

* **Cross-validation** estimates how well the approach generalises. It never
  produces a model you can deploy, and it is the only source of performance
  numbers in this repository.
* **The final model** is trained on the whole cohort and packaged as a bundle
  for inference on new patients. It keeps no validation split of its own:
  epoch counts, ensemble weights, decision thresholds and probability
  calibrators all come from the cross-validation out-of-fold predictions.

### Why the final model has no holdout

Carving a 10% validation set out of 471 patients would cost training data, and
its AUC over ~47 patients would be too noisy to mean anything. Worse, the
ensemble weights are derived from out-of-fold predictions that cover those same
patients, so such a number would also be biased upward. Instead the final model
is simply the evaluated procedure applied to all the data, and every figure
attached to it is the cross-validated one.

## Package layout

```text
src/ais_progression/
|-- config.py         typed configuration; defaults reproduce the paper
|-- evaluation.py     AUC and threshold-dependent metrics
|-- utils.py          seeding, device selection, run metadata
|-- data/             dataset schema, workbook ingestion, preprocessing, loaders
|-- models/           image classifiers (timm + Lightning) and clinical models
|-- ensemble/         weighted averaging and the stacked comparators
|-- experiments/      splitting, repeated nested cross-validation, reporting
|-- final/            full-cohort training, the model bundle, inference
`-- cli/              one module per command
```

## Installation

Python 3.11-3.12 and [uv](https://docs.astral.sh/uv/).

```powershell
git clone https://github.com/ichikawalab/ais-progression-classifier.git
cd ais-progression-classifier
uv sync
```

For NVIDIA GPU training, replace the default torch wheels with a CUDA build
following the [official PyTorch instructions](https://pytorch.org/get-started/locally/).
Verify:

```powershell
uv run python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

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

### Building it from the study workbooks

`ais-build-dataset` joins the clinical workbook with the two image-path
workbooks, re-roots the recorded absolute paths onto the local image
directories, and maps the source `Label` column (2 = progression,
0 = non-progression) to `label`:

```powershell
ais-build-dataset --data-dir data --output-csv data/dataset.csv
```

The workbooks are located by pattern under `--data-dir`, so reorganising the
cohort into subdirectories does not break the command; pass `--clinical-xlsx`,
`--front-xlsx`, `--lateral-xlsx`, `--front-root`, or `--lateral-root` to point
at them explicitly.

Image paths are written relative to the output CSV, so the dataset stays valid
when the cohort is moved or shared. Pass `--absolute-paths` to opt out. It also
writes `data/dataset_report.json` with the cohort summary and any patients
dropped for a missing image or an unusable label.

## Preprocessing

CLAHE followed by zero-padding to a square canvas. Intensity normalisation with
the ImageNet mean and standard deviation happens later, in the training
transform.

```powershell
ais-preprocess --dataset-csv data/dataset.csv --output-dir data/processed --output-csv data/dataset_processed.csv
```

Do not run this twice on the same cohort: CLAHE is not idempotent. DICOM
decoding, windowing, and de-identification are not implemented; convert DICOM
data with a validated local workflow first.

## Cross-validation

Repeated stratified 10-fold cross-validation, 10 repetitions, seeded 42 + r - 1.
Within each repetition one fold is held out for test. Image models carve a
stratified 1/9 slice out of the remaining folds for early stopping, leaving
eight folds' worth of training data. Clinical and ensemble models instead tune
with an inner stratified 10-fold Optuna search over the whole outer training
fold (nested cross-validation).

Run each of the nine individual models:

```powershell
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

```powershell
ais-cv-ensemble --method weighted
ais-cv-ensemble --method average
ais-cv-ensemble --method logreg
ais-cv-ensemble --method svm
ais-cv-ensemble --method rf
```

Each fold is written as it completes, so re-running the same command resumes
where it stopped. Pass `--no-resume` to recompute. Fold weights are discarded
after prediction unless you pass `--keep-checkpoints`; a full run would
otherwise write hundreds of gigabytes.

Every resumable run pins the resolved configuration, cohort table, fold setup,
software versions, Git/source-tree identity, and (for image models) radiograph
bytes in `run_identity.json`. A directory whose identity differs, or which has
folds but no identity, is rejected instead of mixing results from two runs.

To restrict the ensemble to a subset of modalities, name the base models
explicitly:

```powershell
ais-cv-ensemble --method weighted --base front_vit=outputs/cv/front_vit --base clinical_logreg=outputs/cv/clinical_logreg --run-dir outputs/ensemble/front_clinical
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

`summary.json` reports AUC three ways:

* `test_auc_pooled_per_rep` - the headline number. Within a repetition every
  patient has exactly one out-of-fold prediction, so the folds combine into a
  single ROC over the whole cohort; its mean and SD across repetitions are what
  the paper reports.
* `test_auc_per_fold` - test AUC of each individual fold.
* `selection_auc_by_source` - the AUC that drove model selection, grouped by
  where it came from and never pooled across kinds. For image models it is a
  held-out slice (`holdout`); for clinical and ensemble models it is the inner
  cross-validation score (`inner_cv`); for simple averaging it is the training
  fold (`train`). These are not comparable with one another.

Ensemble runs also carry an `ensemble_method_selection_warning`: comparing
several ensemble methods on the same test folds and keeping the best one adds a
separate selection bias, so the winner's AUC should be read as a selected-best
value. The stacking leakage itself is recorded separately.

## Final model

```powershell
ais-train-final --bundle-dir outputs/final
```

This reads the cross-validation runs under `outputs/cv/`, trains every needed
model on all 471 patients, and writes a bundle. Image models train for the
median number of epochs their cross-validated counterparts used, so no data has
to be held back to discover when to stop. The learning-rate schedule still spans
`train.max_epochs`, so those epochs follow the same trajectory as in
cross-validation.

Before training, the command verifies that the selected modality-CV identities
match the current cohort, algorithm, fold settings, software, source tree and
image bytes. Bundle construction happens in a sibling staging directory; the
previous complete bundle is replaced only after the new manifest and artifacts
load successfully.

Image models are stored as bare `state_dict` tensors, which keeps the bundle
small and lets them load with `torch.load(weights_only=True)`. Pass
`--save-full-checkpoints` for full Lightning checkpoints.

### Serving profiles

A bundle can carry several configurations. The decision threshold and the
calibrator belong to a *particular* set of models, so a three-model ensemble
cannot reuse the nine-model threshold without invalidating its reported
sensitivity and specificity. Each profile therefore has its own weights,
threshold, calibrator, and cross-validated AUC — all derived from out-of-fold
predictions, so extra profiles cost no image training. After the outer
cross-validation, one final serving-weight vector is fitted on all base-model
OOF probabilities by maximising the mean full-cohort AUC across repetitions;
it is not an average of the fold-specific weight vectors. The reported AUC,
threshold, and calibrator continue to come from the outer-fold predictions,
not from rescoring the development cohort with those final weights.

The defaults are `full` (every model), `front_clinical`, and `clinical_only`.
Declare your own with `--profile NAME=MODALITIES`:

```powershell
ais-train-final --profile full= --profile cheap=clinical --default-profile full
```

The decision threshold is the median Youden threshold across repetitions. Pass
`--threshold-policy target_sensitivity --target-sensitivity 0.9` to choose the
highest observed threshold whose mean sensitivity across repetitions is at least
0.9. Probabilities are calibrated with isotonic
regression by default (`--calibration platt|none`); calibration is monotonic, so
AUC is unchanged and only the meaning of the number improves.

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

```powershell
ais-predict --bundle-dir outputs/final --input-csv data/new_cases.csv --output-csv predictions.csv
ais-predict --bundle-dir outputs/final --list-profiles
ais-predict --bundle-dir outputs/final --profile clinical_only --input-csv ... --output-csv ...
```

The input CSV needs only the fields used by the selected profile: for example,
`clinical_only` needs no image paths, while a front-only profile needs neither
the lateral path nor clinical variables. `patient_id` is always required and
`label` is optional; when present, the AUC of the ensemble and of each base
model is reported. Output
holds one column per base model plus `probability`, `calibrated_probability`,
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

```powershell
ais-gradcam --bundle-dir outputs/final --modality front --model convnextv2 --input-csv data/dataset.csv --limit 20
```

The paper's three backbones are ViT, Swin and ConvNeXtV2; the target-layer
resolver also covers ResNet, DenseNet, Inception and EfficientNet, so a bundle
built after swapping `image.archs` for one of those still works. Grad-CAM is
exploratory and does not establish a causal explanation for a prediction.

## Configuration

Defaults live in [configs/default.yaml](configs/default.yaml) and reproduce the
published settings: AdamW at lr 1e-5 with weight decay 1e-3, batch size 32, up
to 100 epochs with a 5-epoch linear warmup then cosine annealing, early stopping
after 5 epochs without validation improvement, inverse-frequency class weights,
384x384 inputs, and a shared head of LayerNorm, Linear(512), GELU, Dropout(0.5),
Linear(2). Augmentation is horizontal flipping (p=0.5) and a random resized crop
covering 50-100% of the image at a fixed 1:1 aspect ratio, applied to training
folds only.

Override anything from the command line:

```powershell
ais-cv-modality --modality front --model vit --set train.max_epochs=50 --set data.batch_size=8 --reps 2
```

Mixed-precision training is not bit-reproducible across GPUs. Set
`--set train.precision=32-true` for strict reproducibility; inference always
runs in fp32.

## Tests

```powershell
uv sync --extra dev
uv run pytest -q
uv run ruff check .
```

The integration tests run the whole protocol end to end on a synthetic cohort
with a small CNN, so they finish in seconds on CPU.

## Limitations

- Cross-validation is not external validation.
- The ensemble method was chosen by comparing candidates on the same test folds
  that report its performance, so its AUC is a selected-best value.
- The ensembles are fitted on a single out-of-fold probability matrix, as in the
  published procedure. A training patient's probability therefore came from a
  base model that had seen the current test fold, so the fusion weights are
  chosen with indirect knowledge of it and the reported ensemble AUC is
  optimistic. Removing this would mean regenerating the base models' out-of-fold
  probabilities inside every outer fold -- ten times the image training, and no
  longer the published method.
- The final serving weights are refitted on all base-model OOF probabilities,
  while AUC, threshold and calibration are estimated from outer-fold ensemble
  predictions. As with any full-data refit after cross-validation, those values
  estimate the fitting procedure rather than directly evaluating the exact
  final parameter vector.
- Performance may not generalize across institutions, scanners, or populations.
- Calibration improves how probabilities read, but it is fitted on this cohort
  and carries no guarantee elsewhere.
- Horizontal flipping alters laterality, which may matter for right- and
  left-sided curves; curve direction is not modelled.
- Patients with a 6-9 degree increase were excluded, so the model is untested on
  borderline cases.
- Brace treatment is not a model input, and was not randomly assigned.
- `train.deterministic` is best-effort: operations without a deterministic
  kernel fall back and only warn, and mixed precision is not bit-reproducible
  across GPUs. Use `--set train.precision=32-true` when that matters.

## License and citation

MIT License. See [LICENSE](LICENSE) and [CITATION.cff](CITATION.cff).
