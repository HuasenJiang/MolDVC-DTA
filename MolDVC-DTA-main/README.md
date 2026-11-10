# MolDVC-DTA Training Source

## Snapshot Scope

This source-only snapshot was prepared on 2026-11-10 from the current local
project. The four training entry points were explicitly identified by the author.
Their source and the required local dependencies were copied without modifying
their contents, including line endings. `SNAPSHOT_MANIFEST.json` records each
original source file's SHA-256 checksum and size.

No training, checkpoint inference, dependency installation, or full reproduction
was performed to prepare this snapshot. Read `KNOWN_LIMITATIONS.md` before using
the scripts for a new evaluation. Data and pretrained weights must be obtained and
verified separately; none are bundled in the dedicated source archive.

## Included Source

| File | Role |
| --- | --- |
| `training_for_DTA.py` | Davis/KIBA affinity training; also contains a PDBbind dataset branch |
| `training_for_DTA_LBA.py` | LBA30/LBA60 affinity training with feature-coverage checks |
| `training_for_DTA_PDBbind.py` | Dedicated PDBbind affinity training |
| `training_for_CPI.py` | Five-fold Human interaction classification |
| `models/MolDVC_DTA.py` | Shared MolDVC-DTA model |
| `models/contrastive_learning.py` | Branch contrastive objectives |
| `utils.py` | Datasets, batching, and metrics |
| `create_data_DTA.py` | Graph/sequence preprocessing and `CustomData` definition imported by the trainers |
| `create_data_PDBbind.py` | Existing PDBbind preprocessing utilities |
| `create_data_LBA.py` | Existing LBA split/feature preparation |
| `requirements.txt` | Unmodified historical dependency list; not a complete environment lock |

## Setup And Inputs

Run commands from this directory, not from its parent repository directory:

```bash
cd MolDVC-DTA-main
```

Consult `ENVIRONMENT.md` before installing dependencies. The included historical
`requirements.txt` does not pin PyTorch itself and differs from the environment
previously declared in the repository. No newly tested installation recipe is
claimed here.

Place the required CSV files and compatible `mol_data_M.pkl` / `pro_data_M.pkl`
files under `data/<dataset>/` as specified in `DATA_LAYOUT.md`. Only load pickle
files from trusted sources. Source packaging does not establish the provenance,
coverage, or benchmark-version fidelity of separately obtained data.

## Training Commands

The following commands use flags present in the packaged parsers. They were
checked statically, not executed as training jobs. Select a valid GPU index or use
the scripts' CPU fallback when CUDA is unavailable. Existing results with the
same generated filenames may be overwritten when these commands are run.

Prepare output directories before running trainers that do not create them:

```bash
mkdir -p results/davis results/kiba results/PDBbind results/LBA30 results/LBA60 results/Human
```

```bash
python training_for_DTA.py --dataset davis --device 0 --n_iter 4 --epochs 2000 --batch 1024 --LR 0.0005 --log_interval 20 --num_workers 8
python training_for_DTA.py --dataset kiba --device 0 --n_iter 4 --epochs 2000 --batch 1024 --LR 0.0005 --log_interval 20 --num_workers 8
python training_for_DTA_PDBbind.py --dataset PDBbind --device 0 --n_iter 4 --epochs 2000 --batch 512 --LR 0.0005 --log_interval 20 --num_workers 8
python training_for_DTA_LBA.py --dataset LBA30 --device 0 --n_iter 4 --epochs 2000 --batch 512 --LR 0.0001 --log_interval 20 --num_workers 8
python training_for_DTA_LBA.py --dataset LBA60 --device 0 --n_iter 4 --epochs 2000 --batch 512 --LR 0.0001 --log_interval 20 --num_workers 8
python training_for_CPI.py --dataset Human --device 0 --n_iter 4 --epochs 2000 --batch 512 --LR 0.0005 --num_workers 8
```

`training_for_CPI.py` does not accept `--log_interval`. The regression entry points
also accept `--Alpha`, `--Beta`, and `--window`; CPI accepts `--Alpha` and `--Beta`.
LBA additionally accepts `--feature_dataset` to select a different feature-cache
directory, provided its keys cover all required IDs. `--window` is used in result
filenames; it is not passed to the model constructor in these regression scripts.

These commands describe the supplied source behavior, not a claim that the exact
commands or hyperparameters generated a particular manuscript table.

## Preprocessing

The existing preprocessing files are included for provenance and local imports;
they are not a newly validated, fully portable raw-data pipeline.

- `create_data_DTA.py` accepts `--dataset`, but its default execution does not
  automatically construct every required split or auxiliary file.
- `create_data_PDBbind.py` has a machine-specific path in its main block, not a
  portable `--path` argument. Inspect the code before running it.
- `create_data_LBA.py` accepts explicit raw-data and affinity-table paths. Its
  default split list is `train test`; validation is not processed unless requested.
  It can write/replace caches and filter generated CSVs. Do not run it against a
  benchmark directory without first verifying the data version and preserving
  the original split records.

## Integrity And License

From this directory, verify the listed snapshot files with:

```bash
sha256sum -c SHA256SUMS
```

`PACKAGING_VERIFICATION.json` distinguishes source and syntax checks from runtime
or scientific validation. The inherited MIT license is retained as `LICENSE`;
existing source attribution comments have not been removed. This license does
not grant redistribution rights for separately obtained third-party datasets.
