import argparse
import pickle
from pathlib import Path

import pandas as pd
from tqdm import tqdm


DEFAULT_PDBBIND_RAW_PATH = "/media/ST-18T/huasen/PDBbind/PDBbind_v2020/refined-set"


AA_CODES = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "LYS": "K",
    "ILE": "I",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TYR": "Y",
    "TRP": "W",
}


def read_split_ids(split_file):
    ids = []
    with split_file.open("r", encoding="utf-8") as f:
        for line in f:
            pdb_id = line.strip().lower()
            if pdb_id:
                ids.append(pdb_id)
    duplicate_ids = sorted({pdb_id for pdb_id in ids if ids.count(pdb_id) > 1})
    if duplicate_ids:
        raise ValueError(f"{split_file} contains duplicate ids: {duplicate_ids[:10]}")
    return ids


def load_affinity_table(affinity_file):
    affinity_df = pd.read_csv(affinity_file, dtype={"id": str})
    required_columns = {"id", "affinity"}
    missing_columns = required_columns - set(affinity_df.columns)
    if missing_columns:
        raise ValueError(f"{affinity_file} is missing columns: {sorted(missing_columns)}")

    affinity_df["id_key"] = affinity_df["id"].str.strip().str.lower()
    duplicated = affinity_df[affinity_df["id_key"].duplicated()]["id_key"].unique()
    if len(duplicated) > 0:
        raise ValueError(f"{affinity_file} contains duplicate ids: {duplicated[:10].tolist()}")
    return affinity_df.set_index("id_key")


def build_split_csv(split_file, affinity_table, output_file, drop_missing_ids=False):
    split_ids = read_split_ids(split_file)
    missing_ids = [pdb_id for pdb_id in split_ids if pdb_id not in affinity_table.index]
    if missing_ids and not drop_missing_ids:
        raise ValueError(
            f"{split_file} has {len(missing_ids)} ids missing from the affinity table. "
            f"Examples: {missing_ids[:10]}"
        )
    if missing_ids:
        print(f"Drop {len(missing_ids)} ids from {split_file}. Examples: {missing_ids[:10]}")

    kept_ids = [pdb_id for pdb_id in split_ids if pdb_id in affinity_table.index]
    out_df = pd.DataFrame(
        {
            "num": range(1, len(kept_ids) + 1),
            "id": kept_ids,
            "affinity": [affinity_table.loc[pdb_id, "affinity"] for pdb_id in kept_ids],
        }
    )
    out_df.to_csv(output_file, index=False)
    return out_df


def build_csv_files(dataset_dir, affinity_table, splits, drop_missing_ids):
    all_ids = set()
    for split_name in splits:
        split_file = dataset_dir / f"{split_name}.txt"
        if not split_file.exists():
            print(f"Skip {dataset_dir.name}/{split_name}: {split_file} not found")
            continue

        output_file = dataset_dir / f"{split_name}.csv"
        out_df = build_split_csv(split_file, affinity_table, output_file, drop_missing_ids)
        all_ids.update(out_df["id"])
        print(f"Wrote {output_file}: {len(out_df)} rows")

        if split_name == "val":
            valid_alias = dataset_dir / "valid.csv"
            out_df.to_csv(valid_alias, index=False)
            print(f"Wrote {valid_alias}: {len(out_df)} rows")
    return all_ids


def build_pdb_dir_index(raw_root, ids):
    ids = {pdb_id.lower() for pdb_id in ids}
    dir_index = {}

    for path in raw_root.iterdir():
        if path.is_dir() and path.name.lower() in ids:
            dir_index[path.name.lower()] = path

    missing = ids - set(dir_index)
    if missing:
        for path in raw_root.rglob("*"):
            if not path.is_dir():
                continue
            key = path.name.lower()
            if key in missing and key not in dir_index:
                dir_index[key] = path
                if len(dir_index) == len(ids):
                    break
    return dir_index


def parse_pdb_to_sequence(pdb_file, max_length=1200):
    from Bio.PDB import PDBParser
    from Bio.PDB.Polypeptide import PPBuilder

    structure = PDBParser(QUIET=True).get_structure("protein", str(pdb_file))
    residues = []
    ppb = PPBuilder()
    for chain in structure[0]:
        for peptide in ppb.build_peptides(chain):
            residues.extend(peptide)

    if not residues:
        for chain in structure[0]:
            for residue in chain:
                residues.append(residue)

    sequence = []
    for residue in residues:
        aa = AA_CODES.get(residue.get_resname())
        if aa is None:
            continue
        if "CA" not in residue:
            continue
        sequence.append(aa)
        if len(sequence) >= max_length:
            break

    if not sequence:
        raise ValueError(f"No valid amino-acid sequence extracted from {pdb_file}")
    return "".join(sequence)


def read_ligand_mol(pdb_id, complex_dir):
    from rdkit import Chem
    from create_data_DTA import self_partial_sanitization, self_repair_molecule_file, self_repair_unusual_elements

    mol2_path = complex_dir / f"{pdb_id}_ligand.mol2"
    sdf_path = complex_dir / f"{pdb_id}_ligand.sdf"

    mol = None
    if mol2_path.exists():
        mol = Chem.MolFromMol2File(str(mol2_path), sanitize=False, removeHs=False)

    if mol is None and sdf_path.exists():
        supplier = Chem.SDMolSupplier(str(sdf_path), sanitize=False, removeHs=False)
        if supplier is not None and len(supplier) > 0:
            mol = supplier[0]

    if mol is None:
        mol = self_repair_molecule_file(str(mol2_path), str(sdf_path))

    if mol is None:
        raise ValueError(f"Cannot parse ligand for {pdb_id}")

    try:
        self_repair_unusual_elements(mol)
        self_partial_sanitization(mol)
    except Exception as exc:
        print(f"WARNING: ligand sanitization failed for {pdb_id}: {exc}")
    return mol


def construct_graph_files(dataset_dir, ids, raw_root):
    from create_data_DTA import generate_drug_data, seq_to_feat

    raw_root = Path(raw_root)
    if not raw_root.exists():
        raise FileNotFoundError(f"PDBbind raw directory not found: {raw_root}")

    dir_index = build_pdb_dir_index(raw_root, ids)
    mol_data = {}
    pro_data = {}
    sequences = {}
    failures = []

    for pdb_id in tqdm(sorted(ids), desc=f"Processing {dataset_dir.name} complexes"):
        complex_dir = dir_index.get(pdb_id)
        if complex_dir is None:
            failures.append((pdb_id, "complex directory not found"))
            continue

        try:
            mol = read_ligand_mol(pdb_id, complex_dir)
            mol_data[pdb_id] = generate_drug_data(mol, None)
        except Exception as exc:
            failures.append((pdb_id, f"ligand: {exc}"))
            continue

        protein_file = complex_dir / f"{pdb_id}_protein.pdb"
        if not protein_file.exists():
            failures.append((pdb_id, "protein pdb not found"))
            mol_data.pop(pdb_id, None)
            continue

        try:
            sequence = parse_pdb_to_sequence(protein_file)
            pro_data[pdb_id] = seq_to_feat(sequence)
            sequences[pdb_id] = sequence
        except Exception as exc:
            failures.append((pdb_id, f"protein: {exc}"))
            mol_data.pop(pdb_id, None)
            continue

    with (dataset_dir / "mol_data_M.pkl").open("wb") as f:
        pickle.dump(mol_data, f)
    with (dataset_dir / "pro_data_M.pkl").open("wb") as f:
        pickle.dump(pro_data, f)

    with (dataset_dir / "protein_sequences.txt").open("w", encoding="utf-8") as f:
        for pdb_id, sequence in sorted(sequences.items()):
            f.write(f"{pdb_id}:{sequence}\n")

    if failures:
        pd.DataFrame(failures, columns=["id", "reason"]).to_csv(dataset_dir / "failed_lba_graph_ids.csv", index=False)

    print(f"Wrote {dataset_dir / 'mol_data_M.pkl'}: {len(mol_data)} complexes")
    print(f"Wrote {dataset_dir / 'pro_data_M.pkl'}: {len(pro_data)} complexes")
    if failures:
        print(f"Failed {len(failures)} complexes. See {dataset_dir / 'failed_lba_graph_ids.csv'}")
    return set(mol_data) & set(pro_data)


def filter_csv_files_to_graph_ids(dataset_dir, splits, graph_ids):
    for split_name in splits:
        csv_file = dataset_dir / f"{split_name}.csv"
        if not csv_file.exists():
            continue
        df = pd.read_csv(csv_file, dtype={"id": str})
        before = len(df)
        df["id_key"] = df["id"].str.strip().str.lower()
        df = df[df["id_key"].isin(graph_ids)].copy()
        df = df.drop(columns=["id_key"])
        df["num"] = range(1, len(df) + 1)
        df.to_csv(csv_file, index=False)
        if before != len(df):
            print(f"Filtered {csv_file}: {before} -> {len(df)} rows with graph features")

        if split_name == "val":
            valid_alias = dataset_dir / "valid.csv"
            df.to_csv(valid_alias, index=False)


def main(args):
    data_root = Path(args.data_root)
    affinity_table = load_affinity_table(Path(args.affinity_file))

    dataset_ids = {}
    for dataset in args.datasets:
        dataset_dir = data_root / dataset
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
        ids = build_csv_files(dataset_dir, affinity_table, args.splits, args.drop_missing_ids)
        dataset_ids[dataset] = ids

    if args.skip_graph or args.pdbbind_raw_path is None:
        print("CSV files are ready. Graph generation was skipped.")
        return

    for dataset, ids in dataset_ids.items():
        dataset_dir = data_root / dataset
        graph_ids = construct_graph_files(dataset_dir, ids, Path(args.pdbbind_raw_path))
        filter_csv_files_to_graph_ids(dataset_dir, args.splits, graph_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--affinity_file", type=str, default="data/PDBbind/affinity_data.csv")
    parser.add_argument("--datasets", nargs="+", default=["LBA30", "LBA60"], choices=["LBA30", "LBA60"])
    parser.add_argument("--splits", nargs="+", default=["train", "test"], choices=["train", "val", "test"])
    parser.add_argument("--drop_missing_ids", action="store_true")
    parser.add_argument("--pdbbind_raw_path", type=str, default=DEFAULT_PDBBIND_RAW_PATH)
    parser.add_argument("--skip_graph", action="store_true")
    args = parser.parse_args()
    main(args)
