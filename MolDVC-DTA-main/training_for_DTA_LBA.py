from pathlib import Path
import argparse

from torch import nn
from models.MolDVC_DTA import MolDVC_DTA
from utils import *
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from scipy.stats import pearsonr, spearmanr
from create_data_DTA import CustomData


def train(model, device, train_loader, optimizer, loss_fn, args, epoch):
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output, CL_loss = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device)) + CL_loss
        loss.backward()
        optimizer.step()


def test(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output, _ = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def correlation_metrics(y_true, y_pred):
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan, np.nan
    pearson = pearsonr(y_true, y_pred)[0]
    spearman = spearmanr(y_true, y_pred)[0]
    return pearson, spearman


def ensure_feature_coverage(ids, mol_data, pro_data, split_name, feature_dataset):
    missing_mol = [pdb_id for pdb_id in ids if pdb_id not in mol_data]
    missing_pro = [pdb_id for pdb_id in ids if pdb_id not in pro_data]
    if missing_mol or missing_pro:
        details = []
        if missing_mol:
            details.append(f"mol_data missing {len(missing_mol)} ids, examples: {missing_mol[:10]}")
        if missing_pro:
            details.append(f"pro_data missing {len(missing_pro)} ids, examples: {missing_pro[:10]}")
        raise KeyError(
            f"{split_name} split cannot be trained with data/{feature_dataset}/*.pkl. "
            + "; ".join(details)
        )


def main(args):
    dataset = args.dataset
    feature_dataset = args.feature_dataset or dataset
    model_st = "MolDVC_DTA"
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    df_train = pd.read_csv(f"data/{dataset}/train.csv", dtype={"id": str})
    df_test = pd.read_csv(f"data/{dataset}/test.csv", dtype={"id": str})
    train_id = [pdb_id.lower() for pdb_id in list(df_train["id"])]
    test_id = [pdb_id.lower() for pdb_id in list(df_test["id"])]
    train_label = list(df_train["affinity"])
    test_label = list(df_test["affinity"])

    with open(f"data/{feature_dataset}/mol_data_M.pkl", "rb") as file:
        mol_data = pickle.load(file)
    with open(f"data/{feature_dataset}/pro_data_M.pkl", "rb") as file2:
        pro_data = pickle.load(file2)

    ensure_feature_coverage(train_id, mol_data, pro_data, "train", feature_dataset)
    ensure_feature_coverage(test_id, mol_data, pro_data, "test", feature_dataset)

    train_dataset = PDBDataset(train_id, train_label, mol_data=mol_data, pro_data=pro_data)
    test_dataset = PDBDataset(test_id, test_label, mol_data=mol_data, pro_data=pro_data)
    train_loader = DrugDataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=args.num_workers)
    test_loader = DrugDataLoader(test_dataset, batch_size=args.batch, shuffle=False, num_workers=args.num_workers)



    model = MolDVC_DTA(
        vocab_protein_size=26,
        embedding_size=128,
        out_dim=1,
        Alpha=args.Alpha,
        Beta=args.Beta,
        n_iter=args.n_iter,
    ).to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.LR)
    best_rmse = 1000
    best_epoch = -1
    best_ci = best_pearson = best_spearman = None

    result_dir = Path(f"results/{dataset}")
    result_dir.mkdir(parents=True, exist_ok=True)
    model_file_name = result_dir / f"train_{model_st}_{dataset} + {args.window}.model"
    result_file_name = result_dir / f"train_{model_st}_{dataset} + {args.window}.csv"

    for epoch in range(args.epochs):
        train(model, device, train_loader, optimizer, loss_fn, args, epoch + 1)
        G, P = test(model, device, test_loader)
        pearson, spearman = correlation_metrics(G, P)
        ret = [rmse(G, P), concordance_index(G, P), pearson, spearman]
        if ret[0] < best_rmse:
            torch.save(model.state_dict(), model_file_name)
            with result_file_name.open("w") as f:
                f.write(",".join(map(str, ret)))
            best_epoch = epoch + 1
            best_rmse = ret[0]
            best_ci = ret[1]
            best_pearson = ret[2]
            best_spearman = ret[3]

        if (epoch + 1) % args.log_interval == 0:
            print(
                "Epoch: {} | Best epoch: {} | RMSE: {:.6f} | CI: {:.6f} | Pearson: {:.6f} | Spearman: {:.6f}".format(
                    epoch + 1,
                    best_epoch,
                    best_rmse,
                    best_ci,
                    best_pearson,
                    best_spearman,
                )
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--Alpha", type=float, default=0.01, help="Coefficient of NG_Loss")
    parser.add_argument("--Beta", type=float, default=0.01, help="Coefficient of EG_Loss")
    parser.add_argument("--n_iter", type=int, default=4, help="layers of sub_structure")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--LR", type=float, default=0.0001)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="LBA30", choices=["LBA30", "LBA60"])
    parser.add_argument("--feature_dataset", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--window", type=int, default=3)
    args = parser.parse_args()
    main(args)
