# Snapshot Behavior And Unverified Items

This package intentionally preserves the selected current source. Packaging did
not correct its evaluation protocol or change model behavior.

1. The regression trainers evaluate `test.csv` each epoch and select checkpoints
   by test MSE or test RMSE. The CPI trainer selects by the supplied outer test
   fold's AUROC. These are not validation-only selection procedures; unchanged
   scores must not be described as independently selected held-out test results.
2. Explicit feature-noise sampling remains active in `model.eval()`. Dropout is
   disabled, but evaluations can still vary. These entry points do not provide a
   documented fixed-seed or Monte Carlo inference protocol.
3. The data loader constructs ordinary PyG `Data` objects. Correct offsetting of
   `line_graph_edge_index` across molecules needs verification in the actual PyG
   runtime; merely importing `CustomData` does not establish that its batching
   override is used by the reconstructed samples.
4. Prepared datasets, split IDs, feature caches, checkpoints, and their provenance
   are not included in this source-only archive. Data coverage, cross-split
   overlap, official benchmark versions, and labels need separate verification.
5. Preprocessing contains original machine-specific data paths. They were retained
   to preserve the author's source; they are not portable download locations.
   LBA preprocessing can overwrite feature caches and filter split CSVs.
6. Syntax and checksum checks are not dependency, training, performance, or
   checkpoint-compatibility tests. Existing weights retained in the GitHub
   repository have not been matched to this snapshot by this packaging step.

These limitations should be resolved or disclosed as appropriate before making
new reproducibility or comparative-performance claims. This release does not
claim that the above issues affected every historical experiment or quantify
their effect on any manuscript result.
