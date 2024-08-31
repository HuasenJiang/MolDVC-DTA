# -*- coding:utf-8 -*-
from models.MolDVC_DTA import MolDVC_DTA
from torch import nn
from utils import *
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, precision_score, recall_score
import argparse
from create_data_DTA import CustomData

def train(model, device, train_loader,optimizer,loss_fn,epoch):
    model.train()
    epoch_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output,CL_loss= model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device)) + CL_loss
        loss.backward()
        optimizer.step()
        current_batch_size = len(data.y)
        epoch_loss += loss.item() * current_batch_size
    print('Epoch {}: train_loss: {:.5f} '.format(epoch, epoch_loss / len(train_loader.dataset)), end='')

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
    with open(f"data/{dataset}/mol_data_M.pkl", 'rb') as file:
        mol_data = pickle.load(file)
    with open(f'data/{dataset}/pro_data_M.pkl', 'rb') as file2:
        pro_data = pickle.load(file2)

    results = []
    for fold in range(1,6): # Five-fold cross-training
        df_train = pd.read_csv(f'data/{dataset}/train{fold}.csv') #Reading training data for current fold.
        df_test = pd.read_csv(f'data/{dataset}/test{fold}.csv') #Reading test data for current fold.
        train_smile,train_seq,train_label = list(df_train['compound_iso_smiles']), list(df_train['target_sequence']),list(df_train['affinity'])
        test_smile,test_seq,test_label = list(df_test['compound_iso_smiles']), list(df_test['target_sequence']),list(df_test['affinity'])
        train_dataset = CPIDataset(train_smile, train_seq, train_label, mol_data = mol_data, pro_data=pro_data)
        test_dataset = CPIDataset(test_smile, test_seq, test_label, mol_data = mol_data, pro_data=pro_data)
        train_loader = DrugDataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=args.num_workers)
        test_loader = DrugDataLoader(test_dataset, batch_size=args.batch, shuffle=False, num_workers=args.num_workers)

        # training the model
        model = MolDVC_DTA(vocab_protein_size = 26,embedding_size=128, out_dim=1,Alpha=args.Alpha,Beta=args.Beta,n_iter=args.n_iter).to(device)
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.LR)
        best_roc = 0
        best_epoch = -1
        model_file_name = f'results/{dataset}/' + model_st + '_'+ dataset + '_fold' + str(fold) + '.model'
        for epoch in range(args.epochs):
            train(model, device, train_loader, optimizer, loss_fn, epoch + 1)
            G, P ,_= test(model, device, test_loader)
            valid_roc = roc_auc_score(G, P)
            print('| AUROC: {:.5f}'.format(valid_roc))
            if valid_roc > best_roc:
                best_roc = valid_roc
                best_epoch = epoch + 1
                torch.save(model.state_dict(), model_file_name)  #Saving the model parameters with the best performance during training.
                tpr, fpr, _ = precision_recall_curve(G, P)
                ret = [roc_auc_score(G, P), auc(fpr, tpr)]
                test_roc = ret[0]
                test_prc = ret[1]
                print('AUROC improved at epoch ', best_epoch, '; test_auc:{:.5f}'.format(test_roc),
                      '; test_prc:{:.5f}'.format(test_prc), model_st, dataset)
            else:
                print('No improvement since epoch ', best_epoch, '; test_auc:{:.5f}'.format(test_roc),
                      '; test_prc:{:.5f}'.format(test_prc), model_st, dataset)

        # reload the best model and test it on valid set again to get other metrics
        model.load_state_dict(torch.load(model_file_name))
        G, P_value, P_label = test(model, device, test_loader)
        tpr, fpr, _ = precision_recall_curve(G, P_value)
        valid_metrics = [roc_auc_score(G, P_value), auc(fpr, tpr), precision_score(G, P_label),recall_score(G, P_label)]
        print('Fold-{} valid finished, auc: {:.5f} | prc: {:.5f} | precision: {:.5f} | recall: {:.5f}'.format(str(fold),valid_metrics[0],valid_metrics[1],valid_metrics[2],valid_metrics[3]))
        results.append([valid_metrics[0], valid_metrics[1], valid_metrics[2], valid_metrics[3]])
    valid_results = np.array(results)
    valid_results = [np.mean(valid_results, axis=0), np.std(valid_results, axis=0)]
    print("5-fold cross validation finished. " "auc:{:.3f}±{:.4f} | prc:{:.3f}±{:.4f} | precision:{:.3f}±{:.4f} | recall:{:.3f}±{:.4f}".format(valid_results[0][0], valid_results[1][0], valid_results[0][1], valid_results[1][1],valid_results[0][2], valid_results[1][2], valid_results[0][3], valid_results[1][3]))
    result_file_name = f'results/{dataset}/' + model_st +'_'+ dataset + '.txt'  # result
    with open(result_file_name, 'w') as f:
        f.write("auc:{:.3f}±{:.4f} | prc:{:.3f}±{:.4f} | precision:{:.3f}±{:.4f} | recall:{:.3f}±{:.4f}".format(valid_results[0][0], valid_results[1][0], valid_results[0][1], valid_results[1][1],valid_results[0][2], valid_results[1][2], valid_results[0][3], valid_results[1][3]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Alpha', type = float, default = 0.01, help='Coefficient of NG_Loss')
    parser.add_argument('--Beta',  type = float, default = 0.01, help='Coefficient of EG_Loss')
    parser.add_argument('--n_iter',type = int, default = 4, help='layers of sub_structure')
    parser.add_argument('--epochs', type = int, default = 2000)
    parser.add_argument('--batch', type = int, default = 512)
    parser.add_argument('--LR', type = float, default = 0.0005)
    parser.add_argument('--device', type = int, default = 0)
    parser.add_argument('--dataset', type = str, default = 'Human')
    parser.add_argument('--num_workers', type= int, default = 8)
    args = parser.parse_args()
    main(args)







