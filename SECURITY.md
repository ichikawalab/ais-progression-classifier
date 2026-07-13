# Security policy

Report security issues privately to the repository maintainers rather than in a
public issue.

Lightning `.ckpt` files may contain pickle data. Load only checkpoints produced
by a trusted local training run. Do not download and open an untrusted
checkpoint. Public model releases should use a weights-only format such as
`safetensors` and publish a checksum.

Do not include patient data, protected health information, credentials, or
institutional filesystem paths in bug reports.

The public repository must not contain patient-derived data, medical images,
DICOM files, cohort tables, trained weights, checkpoints, predictions, or
experiment outputs. Only synthetic CSV files under `examples/` are permitted.
