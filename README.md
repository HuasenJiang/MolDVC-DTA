# MolDVC-DTA

The project source is in [`MolDVC-DTA-main/`](MolDVC-DTA-main/).

The 2026-11-10 training-code snapshot contains these author-designated entry points:

- `training_for_DTA.py`: Davis and KIBA affinity prediction.
- `training_for_DTA_LBA.py`: LBA30 and LBA60 affinity prediction.
- `training_for_DTA_PDBbind.py`: PDBbind affinity prediction.
- `training_for_CPI.py`: Human interaction classification.

See the [current training instructions](MolDVC-DTA-main/README.md),
[data requirements](MolDVC-DTA-main/DATA_LAYOUT.md),
[dependency notes](MolDVC-DTA-main/ENVIRONMENT.md), and
[snapshot limitations](MolDVC-DTA-main/KNOWN_LIMITATIONS.md).

The snapshot preserves the selected local source files byte for byte; it is not a
new training run or a claim that all manuscript results have been reproduced.
Dataset files, feature caches, and model weights are not included in the dedicated
source ZIP. Existing repository weights are retained separately, without a new
compatibility claim.

On GitHub, the dedicated source archive is in [`code_snapshots/`](code_snapshots/).
The snapshot manifest, verification report, and SHA-256 checksums are under
`MolDVC-DTA-main/`. The existing [MIT license](MolDVC-DTA-main/LICENSE) is retained.
