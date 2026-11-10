# Dependency Notes

The packaged `requirements.txt` is the existing local file, preserved byte for
byte. It is historical, not a newly generated or tested environment lock.

The four training entry points and their transitive local imports require:

- PyTorch and PyTorch Geometric.
- NumPy, pandas, SciPy, and scikit-learn.
- lifelines for the regression concordance metric.
- RDKit, NetworkX, and tqdm, including imports reached through `create_data_DTA`.
- Biopython when using the PDBbind/LBA structure-preparation utilities.

The historical list includes `torch-geometric==2.0.4`, but it does not pin
`torch` itself and does not include Biopython. It also contains older torchvision
and torchaudio pins. Do not assume that installing it into an arbitrary current
Python/CUDA environment produces a working training runtime.

For reference only, the previously published repository README declared Python
3.8, PyTorch 2.0.1, CUDA 11.8, PyG 2.4.0, RDKit 2023.9.5, pandas 2.0.3, and
scikit-learn 1.3.2. Those declared versions were not installed, tested, or confirmed
as the exact environment of the author's final checkpoint during this packaging
step. The historical requirements were not silently replaced with that list.

Before a reproducibility release, record the versions actually used for training,
matching PyTorch/PyG extension builds, Python/CUDA versions, device information,
and any serialization constraints. Verify imports and a small data/model pass in
that environment, then freeze the validated configuration. No such runtime test
or environment certification is claimed by this source archive.
