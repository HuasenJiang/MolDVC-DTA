from models.MolDVC_DTA import MolDVC_DTA
from utils import *
import pandas as pd
from lifelines.utils import concordance_index
import argparse
from create_data_DTA import CustomData


def test(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output,_ = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0) #predicted values
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0) #ground truth
    return total_labels.numpy().flatten(),total_preds.numpy().flatten()

def main(args):
    dataset = args.dataset
    model_st = 'MolDVC_DTA'
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else "cpu")

    path = f'results/{dataset}/train_MolDVC_DTA_kiba.model'
    check_point = torch.load(path,map_location=device)
    model = MolDVC_DTA(vocab_protein_size = 26,embedding_size=128,out_dim=1,Alpha=args.Alpha,Beta=args.Beta,n_iter=args.n_iter)
    model.load_state_dict(check_point) # Loading pre-trained model parameters into the current model.

    model = model.to(device)
    df_test = pd.read_csv(f'data/{dataset}/test.csv') # Reading test data.
    test_smile, test_seq, test_label = list(df_test['compound_iso_smiles']), list(df_test['target_sequence']), list(df_test['affinity'])

    with open(f"data/{dataset}/mol_data_M.pkl", 'rb') as file:
        mol_data = pickle.load(file) # Reading drug graph data from the serialized file.
    with open(f'data/{dataset}/pro_data_M.pkl', 'rb') as file2:
        pro_data = pickle.load(file2) # Reading protein graph data from the serialized file.
    test_dataset = DTADataset(test_smile, test_seq, test_label, mol_data = mol_data, pro_data = pro_data)
    test_loader = DrugDataLoader(test_dataset, batch_size=args.batch, shuffle=False, num_workers=args.num_workers)

    G, P = test(model, device, test_loader)
    G_list = G.tolist()
    P_list = P.tolist()
    predicted_data = {
        'smile':test_smile,
        'sequence':test_seq,
        'label':G_list,
        'predicted value':P_list
    }
    df_pre = pd.DataFrame(predicted_data)
    df_pre.to_csv(f'./results/{args.dataset}/predicted_value_of_{model_st}_on_{args.dataset} .csv') #Record the prediction results to a CSV file.
    ret = [mse(G, P),concordance_index(G, P)]
    print(args.dataset, model_st,'test_mse:', ret[0], 'test_ci:',ret[1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Alpha', type = float, default = 0.01, help='Coefficient of NG_Loss')
    parser.add_argument('--Beta',  type = float, default = 0.01, help='Coefficient of LG_Loss')
    parser.add_argument('--n_iter',type = int, default = 4, help='layers of sub_structure')
    parser.add_argument('--batch', type = int, default = 512)
    parser.add_argument('--device', type = int, default = 0)
    parser.add_argument('--dataset', type = str, default = 'davis',choices=['davis','kiba'])
    parser.add_argument('--num_workers', type= int, default = 8)
    args = parser.parse_args()
    main(args)
