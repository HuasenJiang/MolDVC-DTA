from models.MolDVC_DTA import MolDVC_DTA
from utils import *
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, precision_score, recall_score
import argparse
from create_data_DTA import CustomData

def test(model, device, loader):
    model.eval()
    total_pred_values = torch.Tensor()
    total_pred_labels = torch.Tensor()
    total_true_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output,_ = model(data)
            predicted_values = torch.sigmoid(output)  # continuous
            predicted_labels = torch.round(predicted_values)  # binary
            total_pred_values = torch.cat((total_pred_values, predicted_values.cpu()), 0)  # continuous
            total_pred_labels = torch.cat((total_pred_labels, predicted_labels.cpu()), 0)  # binary
            total_true_labels = torch.cat((total_true_labels, data.y.view(-1, 1).cpu()), 0)
    return total_true_labels.numpy().flatten(), total_pred_values.numpy().flatten(), total_pred_labels.numpy().flatten()

def main(args):
    dataset = args.dataset
    model_st = 'MolDVC_DTA'
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else "cpu")

    results = []
    for fold in range(1,6): # Test the performance of the model trained with five-fold cross-training sequentially.
        df_test = pd.read_csv(f'data/{dataset}/test{fold}.csv')  # Reading test data.
        test_smile, test_seq, test_label = list(df_test['compound_iso_smiles']), list(df_test['target_sequence']), list(
            df_test['affinity'])

        with open(f"data/{dataset}/mol_data_M.pkl", 'rb') as file:
            mol_data = pickle.load(file)  # Reading drug graph data from the serialized file.
        with open(f'data/{dataset}/pro_data_M.pkl', 'rb') as file2:
            pro_data = pickle.load(file2)  # Reading protein graph data from the serialized file.
        test_dataset = CPIDataset(test_smile, test_seq, test_label, mol_data=mol_data, pro_data=pro_data)
        test_loader = DrugDataLoader(test_dataset, batch_size=args.batch, shuffle=False, num_workers=args.num_workers)

        model = MolDVC_DTA(vocab_protein_size=26, embedding_size=128, out_dim=1, Alpha=args.Alpha, Beta=args.Beta,
                           n_iter=args.n_iter)
        model = model.to(device)
        model_file_name = f'results/{dataset}/train_MolDVC_DTA_Human_fold{fold}.model'
        model.load_state_dict(torch.load(model_file_name)) # Loading pre-trained model parameters into the current model.
        G, P_value, P_label = test(model, device, test_loader)

        G_list = G.tolist()
        P_value_list = P_value.tolist()
        P_label_list = P_label.tolist()
        predicted_data = {
            'smile': test_smile,
            'sequence': test_seq,
            'label': G_list,
            'predicted value': P_value_list,
            'predicted label': P_label_list
        }
        df_pre = pd.DataFrame(predicted_data)
        df_pre.to_csv(f'./results/{args.dataset}/predicted_value_of_{args.model}_on_{args.dataset}_test{fold} .csv')
        tpr, fpr, _ = precision_recall_curve(G, P_value)
        valid_metrics = [roc_auc_score(G, P_value), auc(fpr, tpr), precision_score(G, P_label),recall_score(G, P_label)]
        print('Fold-{}: prc: {:.5f} | auc: {:.5f} | precision: {:.5f} | recall: {:.5f}'.format(str(fold),valid_metrics[0],valid_metrics[1],valid_metrics[2],valid_metrics[3]))
        results.append([valid_metrics[0], valid_metrics[1], valid_metrics[2], valid_metrics[3]])

    valid_results = np.array(results)
    #Calculating the average performance of all models.
    valid_results = [np.mean(valid_results, axis=0), np.std(valid_results, axis=0)]
    print("5-fold results:" "prc:{:.3f}±{:.4f} | auc:{:.3f}±{:.4f} | precision:{:.3f}±{:.4f} | recall:{:.3f}±{:.4f}".format(valid_results[0][0], valid_results[1][0], valid_results[0][1], valid_results[1][1],valid_results[0][2], valid_results[1][2], valid_results[0][3], valid_results[1][3]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Alpha', type = float, default = 0.01, help='Coefficient of NG_Loss')
    parser.add_argument('--Beta',  type = float, default = 0.01, help='Coefficient of EG_Loss')
    parser.add_argument('--n_iter',type = int, default = 4, help='layers of sub_structure')
    parser.add_argument('--batch', type = int, default = 512)
    parser.add_argument('--device', type = int, default = 0)
    parser.add_argument('--dataset', type = str, default = 'Human')
    parser.add_argument('--num_workers', type= int, default = 8)
    args = parser.parse_args()
    main(args)
