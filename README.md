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
