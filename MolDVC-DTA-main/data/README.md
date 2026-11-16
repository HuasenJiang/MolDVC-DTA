# Processed Representations And Splits

The authors provide molecular and protein representations for Davis, KIBA, and
Human CPI, together with the original split CSVs consumed by the current default
training entry points:

```text
data/
  davis/mol_data_M.pkl
  davis/pro_data_M.pkl
  davis/train.csv
  davis/test.csv
  kiba/mol_data_M.pkl
  kiba/pro_data_M.pkl
  kiba/train.csv.gz
  kiba/test.csv
  Human/mol_data_M.pkl
  Human/pro_data_M.pkl
  Human/train1.csv ... train5.csv
  Human/test1.csv  ... test5.csv
```

These files were copied without changes from the author's local
`data/<dataset>/` folders: six PKLs and 14 CSVs. No preprocessing, training, new
prediction, or pickle deserialization was performed for this upload. CSV line
endings were not normalized. The large KIBA training CSV is stored as
`train.csv.gz`; lossless decompression restores the original `train.csv` exactly.
The original 2026-11-10 source ZIP remains code-only; these files are a separate
addition to the repository.

## Use And Integrity

Clone the repository or download the individual files while preserving their
names and dataset folders. Run trainers from `MolDVC-DTA-main/`. The exact
capitalization of `Human` and lowercase spelling of `davis` and `kiba` matter on
case-sensitive filesystems.

From `MolDVC-DTA-main/data/`, check the files with:

```bash
sha256sum -c SHA256SUMS
```

From `MolDVC-DTA-main/`, prepare KIBA's training input with:

```bash
gzip -dk data/kiba/train.csv.gz
```

Keep the `.gz` file to retain the checksum-verifiable distributed artifact. The
manifest also records the uncompressed CSV size and SHA-256 hash. Decompression
was checked against the source bytes before upload.

`FEATURE_CACHE_MANIFEST.json` records byte sizes, SHA-256 hashes, and Git blob
identifiers for the original 20 files, plus CSV row counts and columns.
Matching these values verifies file
identity, not scientific performance, compatibility with every dependency
version, or the provenance of a historical training run.

Use both the PKLs and the included split CSVs as described in
[`DATA_LAYOUT.md`](../DATA_LAYOUT.md). The current regression trainer reads
`data/<dataset>/train.csv` and `test.csv`; the Human trainer reads all five
`trainN.csv`/`testN.csv` pairs. Additional cold-start directories and the alternative
Davis `5 fold/` directory are not part of this default-input release. Source
dataset links are in the [repository README](../../README.md#data-availability).
Do not alter CSV SMILES or sequence keys independently of the caches. The Human
CSV pairs each contain 5,821 rows (not the 6,728-pair source-benchmark count); row
counts per file are recorded without inferring an unverified filtering rule.

Pickle deserialization can execute code. Only load trusted files, preserve the
original cached class imports, and use a compatible PyTorch/PyG environment.
This release does not change the training or preprocessing code to adapt old
cache formats.

## Sources And Scope

- Davis: [DeepDTA benchmark distribution](https://github.com/hkmztrk/DeepDTA/tree/master/data/davis).
- KIBA: [DeepDTA benchmark distribution](https://github.com/hkmztrk/DeepDTA/tree/master/data/kiba).
- Human CPI: [CPI_prediction benchmark distribution](https://github.com/masashitsubaki/CPI_prediction/tree/master/dataset/human/original).

Cite the original datasets and the distributions used. The repository's MIT
software license does not replace applicable third-party dataset terms.

PDBbind, LBA30, and LBA60 caches were explicitly excluded by the author pending
review of redistribution permissions. They are not included in this cache
release or in its manifest/checksum list. No DrugBank source files, raw structure
files, or unrelated files from the local `data/` directory are included.
