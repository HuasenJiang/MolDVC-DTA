# Required Data Layout

This is a code-only snapshot. None of the inputs below is included in its ZIP.
Resolve data access and redistribution terms separately. The presence of a
preprocessing script does not mean the full raw-data pipeline has been tested.

## Davis And KIBA

For each dataset name, use its exact lowercase spelling:

```text
data/davis/                 data/kiba/
  train.csv                  train.csv
  test.csv                   test.csv
  mol_data_M.pkl             mol_data_M.pkl
  pro_data_M.pkl             pro_data_M.pkl
```

The CSV columns consumed by the trainer are `compound_iso_smiles`,
`target_sequence`, and `affinity`. Ligand caches are keyed by the exact SMILES
strings and protein caches by the exact sequence strings used in the CSVs.

## PDBbind And LBA

```text
data/PDBbind/               data/LBA30/               data/LBA60/
  train.csv                  train.csv                 train.csv
  test.csv                   test.csv                  test.csv
  mol_data_M.pkl             mol_data_M.pkl            mol_data_M.pkl
  pro_data_M.pkl             pro_data_M.pkl            pro_data_M.pkl
```

The dedicated trainers consume CSV columns `id` and `affinity`. LBA lowercases
the loaded IDs; make sure cache keys match that convention. LBA can load caches
from `data/<feature_dataset>/` via `--feature_dataset`.

LBA preprocessing additionally expects split-ID text files such as `train.txt`,
`val.txt`, and `test.txt`, an affinity table, and compatible ligand/protein raw
structures. These training entry points do not consume validation CSVs for model
selection. Retain official validation records separately rather than interpreting
this source snapshot as an implementation of validation-only selection.

The LBA preprocessing defaults refer to a PDBbind v2020 structure location and a
PDBbind affinity table. Do not assume those defaults reproduce an official
Atom3D/PDBbind v2019 benchmark without an explicit ID, label, structure, and
exclusion audit.

## Human CPI

```text
data/Human/
  train1.csv ... train5.csv
  test1.csv  ... test5.csv
  mol_data_M.pkl
  pro_data_M.pkl
```

Each fold CSV uses `compound_iso_smiles`, `target_sequence`, and `affinity`.
The latter contains binary labels for the BCE-with-logits objective. The exact
SMILES/sequence cache-key conventions are the same as for Davis/KIBA. The trainer
reads all five fold pairs; a single `train.csv`/`test.csv` pair is insufficient.
Audit duplicate pairs and cross-split overlap before treating folds as independent
held-out evaluations.

## Cache Object Contract

For all four entry points, the ligand cache values expose `x`, `edge_index`,
`edge_attr`, `line_graph_edge_index`, and `c_size`. Protein values are unpacked as
`(size, seq_feat)`. The model's input dimensions are 78 atom features and 6 bond
features. Cached class paths, PyTorch/PyG compatibility, and batched index offsets
must be checked in the actual training environment.

Only unpickle trusted files. Pickle deserialization can execute code. Cache
compatibility and coverage were not verified as part of building this code-only
archive. No dataset download link was independently validated in this packaging
step.
