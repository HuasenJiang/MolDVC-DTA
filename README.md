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
source ZIP. The author-designated Davis checkpoint is distributed separately
as described below. No KIBA checkpoint is distributed in the current branch.

On GitHub, the dedicated source archive is in [`code_snapshots/`](code_snapshots/).
The snapshot manifest, verification report, and SHA-256 checksums are under
`MolDVC-DTA-main/`. The existing [MIT license](MolDVC-DTA-main/LICENSE) is retained.

## Pretrained Davis Checkpoint

The author-designated Davis checkpoint is available as
[`MolDVC_DTA_davis`](MolDVC-DTA-main/results/davis/MolDVC_DTA_davis)
(with no filename extension). It was trained with `training_for_DTA.py` and
replaces the previously published Davis file. The old KIBA checkpoint has been
removed from the current branch.

The published file is an unchanged copy of the author-selected checkpoint,
renamed only for distribution. It contains a PyTorch model state dictionary.

- Size: 9,732,027 bytes.
- SHA-256: `bd913fe7d3fc26502085d56bfdf8e2c1d5f87de5057660b0943b689d5de7c651`.

The metrics supplied by the author for this individual checkpoint are:

| MSE | RMSE | MAE | CI | r_m^2 |
| --- | --- | --- | --- | --- |
| 0.20485835 | 0.4526128 | 0.22435944 | 0.901940030627468 | 0.7208173259653013 |

These values describe one checkpoint, not a five-seed mean. Publication checks
confirmed file integrity and strict state-dictionary loading into the current
four-layer MolDVC-DTA architecture on CPU, with no missing or unexpected keys.
No training or benchmark inference was performed as part of this upload.

From `MolDVC-DTA-main/`, load the trusted checkpoint into an instantiated,
matching model using:

```python
state = torch.load("results/davis/MolDVC_DTA_davis",
                   map_location="cpu", weights_only=True)
model.load_state_dict(state, strict=True)
```

The existing `test_for_DTA.py` has a legacy hard-coded checkpoint path and is
unchanged in this update. Its old path should not be used as the download
location for the newly published model.

## Data Availability

Public benchmark inputs should be obtained from the upstream providers below,
rather than from a duplicate upload in this repository. These links identify
source datasets; they do not by themselves specify this study's processed inputs,
cold-start partitions, or Human cross-validation folds. See the
[required file layout](MolDVC-DTA-main/DATA_LAYOUT.md) before running a trainer.

### Public Benchmark Downloads

| Dataset | Public source and download | Study setting |
| --- | --- | --- |
| Davis | [DeepDTA benchmark files](https://github.com/hkmztrk/DeepDTA/tree/master/data/davis) | Drug-target affinity regression; 30,056 pairs |
| KIBA | [DeepDTA benchmark files](https://github.com/hkmztrk/DeepDTA/tree/master/data/kiba) | Drug-target affinity regression; 118,254 pairs |
| Human CPI | [CPI_prediction original data](https://github.com/masashitsubaki/CPI_prediction/tree/master/dataset/human/original) | Compound-protein interaction classification; the manuscript reports 6,728 pairs |
| LBA30 and LBA60 | [ATOM3D task description](https://www.atom3d.ai/lba.html); [dataset and split indices, DOI: 10.5281/zenodo.4914718](https://doi.org/10.5281/zenodo.4914718) | Two sequence-identity split protocols derived from PDBbind v2019, distinct from the PDBbind v2020 evaluation |

The ATOM3D deposit provides the 30% and 60% split datasets and their original
train/validation/test indices. Follow its supplied license; an upstream public
download is not permission to redistribute modified data under this project's
MIT license. Cite the original datasets and the upstream distribution used.

### Processed Representations And Splits

The authors provide the following processed representations separately from the
2026-11-10 code-only ZIP. The six files are byte-for-byte copies of
`mol_data_M.pkl` and `pro_data_M.pkl` from the corresponding local dataset folders;
they have not been regenerated, deserialized, or modified for this release.
The 14 original CSV files read by the current default training entry points are
also provided: one train/test pair each for Davis and KIBA, and five pairs for
Human. CSV bytes and line endings are preserved. Only KIBA's large training CSV
is stored using lossless gzip compression; decompress it before training.

| Dataset | Molecular representation | Protein representation | Training/test splits |
| --- | --- | --- | --- |
| Davis | [mol_data_M.pkl](MolDVC-DTA-main/data/davis/mol_data_M.pkl) | [pro_data_M.pkl](MolDVC-DTA-main/data/davis/pro_data_M.pkl) | [train.csv](MolDVC-DTA-main/data/davis/train.csv) / [test.csv](MolDVC-DTA-main/data/davis/test.csv) |
| KIBA | [mol_data_M.pkl](MolDVC-DTA-main/data/kiba/mol_data_M.pkl) | [pro_data_M.pkl](MolDVC-DTA-main/data/kiba/pro_data_M.pkl) | [train.csv.gz](MolDVC-DTA-main/data/kiba/train.csv.gz) / [test.csv](MolDVC-DTA-main/data/kiba/test.csv) |
| Human CPI | [mol_data_M.pkl](MolDVC-DTA-main/data/Human/mol_data_M.pkl) | [pro_data_M.pkl](MolDVC-DTA-main/data/Human/pro_data_M.pkl) | [All five train/test fold pairs](MolDVC-DTA-main/data/Human/) |

Keep these filenames under `MolDVC-DTA-main/data/<dataset>/` when downloading
individual files. A normal repository clone also retrieves them. See the
[cache release notes](MolDVC-DTA-main/data/README.md),
[file manifest](MolDVC-DTA-main/data/FEATURE_CACHE_MANIFEST.json), and
[SHA-256 checksums](MolDVC-DTA-main/data/SHA256SUMS).

From `MolDVC-DTA-main/`, expand KIBA's training input without deleting the archive:

```bash
gzip -dk data/kiba/train.csv.gz
```

The released Human fold CSVs each contain 5,821 rows in total; their training and
test counts are recorded in the manifest. These processed-fold counts should
be distinguished from the 6,728-pair source-benchmark statistic in the manuscript.

Feature caches and split CSVs are separate inputs; both are required. This
release includes the default Davis/KIBA partitions and all five Human folds,
not the additional cold-start directories or alternative Davis five-fold files.
Preserve the exact SMILES/sequence keys and the original cached class imports;
use a compatible PyTorch/PyG environment. Only deserialize trusted pickle files,
since loading a pickle can execute code. Byte checksums establish file integrity,
not a new validation of training results or all feature-key coverage.

PDBbind, LBA30, and LBA60 feature caches are intentionally excluded from this
release, as confirmed by the author pending redistribution-permission review.

### Provider-Managed Data

| Resource | Official access | Access and version notes |
| --- | --- | --- |
| PDBbind v2020 | [PDBbind official platform](https://www.pdbbind-plus.org.cn/); [license agreement](https://www.pdbbind-plus.org.cn/licenseagreement) | Obtain the study's v2020 release under the applicable provider agreement. Do not silently replace it with a newer release. This study uses a 14,127-sample subset, not the entire database. |
| DrugBank approved-drug screening library | [DrugBank releases](https://go.drugbank.com/releases); [academic access](https://go.drugbank.com/academic_research); [license FAQ](https://dev.drugbank.com/guides/faqs) | The study screened 2,639 drugs against 88 eligible targets. Confirm the historical source release and its license. The official download page reported academic downloads temporarily paused when checked on 2026-11-16; these are not currently advertised here as anonymous direct downloads. |

PDBbind structures, affinity labels, DrugBank source exports, and feature caches
derived from them are not redistributed in the new research-output archive while
their applicable redistribution permissions remain unconfirmed. The same review
applies to modified LBA data. A conversion to CSV, pickle, or feature tensors does
not replace the upstream license. Protein accessions can be resolved through
[UniProt](https://www.uniprot.org/); receptor structures used for docking must
additionally be identified by their actual source and structure version.

### Study-Generated Outputs And Zenodo

Study-generated results are being prepared for a separate Zenodo deposit. The
initial local package contains the reported result cells, figure source tables,
Grad-AAM perturbation outputs, target-level enrichment results, LBA split counts,
analysis scripts, and the author-confirmed Davis checkpoint. It also records
checksums of locally available inputs without bundling the upstream datasets.
It is not a complete raw-data or five-run training-log deposit.

**Deposit status: not yet uploaded; no Zenodo DOI has been assigned.** The DOI
and public download link will be added here after the record is deposited and
published. Do not interpret an archive preparation or a reserved DOI as an
already public data release.
