from torch import nn
from models.MolDVC_DTA import MolDVC_DTA
from utils import *
import pandas as pd
from lifelines.utils import concordance_index
import argparse
from create_data_DTA import CustomData
def train(model, device, train_loader, optimizer, loss_fn, args, epoch):
    """
    Training function, which records the training-related logic.
    :param model: The model that we aim to train.
    :param device: The GPU device selected to train our model
    :param train_loader: The dataloader for train dataset
    :param optimizer: Adam optimizer
    :param ppi_adj: The adjacency matrix of the PPI network. Note that the adjacency matrix here is sparse, with dimensions of [2, E].
    :param ppi_features: The feature matrix of the PPI network.
    :param pro_graph: Protein graph data that encompasses all proteins within the dataset.
    :param loss_fn: MSEloss.
    :param args: The parameter namespace object.
    :param epoch: Train epoch
    :return: None
    """
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output,CL_loss = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device)) + CL_loss
        # loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * args.batch,
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))

# def test(model, device, loader,ppi_adj,ppi_features,pro_graph):
def test(model, device, loader):
    '''
        :param model: A model used to predict the binding affinity.
        :param device: Device for loading models and data.
        :param loader: Dataloader used to batch the input data.
        :param ppi_adj: The adjacency matrix of a Protein-Protein Interaction (PPI) graph.
        :param ppi_features: The feature matrix of a Protein-Protein Interaction (PPI) graph.
        :param pro_graph: Graph data encapsulated by all proteins in the current dataset.
        :return: Ground truth and predicted values
        '''
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

    df_train = pd.read_csv(f'data/{dataset}/train.csv')# Reading training data.
    df_test = pd.read_csv(f'data/{dataset}/test.csv') # Reading test data.
    with open(f"data/{dataset}/mol_data_M.pkl", 'rb') as file:
        mol_data = pickle.load(file) # Reading drug graph data from the serialized file.
    with open(f'data/{dataset}/pro_data_MM.pkl', 'rb') as file2:
        pro_data = pickle.load(file2) # Reading protein graph data from the serialized file.


    if dataset in ['davis','kiba']:
        train_smile,train_seq,train_label = list(df_train['compound_iso_smiles']), list(df_train['target_sequence']),list(df_train['affinity'])
        test_smile,test_seq,test_label = list(df_test['compound_iso_smiles']), list(df_test['target_sequence']),list(df_test['affinity'])
        train_dataset = DTADataset(train_smile, train_seq, train_label, mol_data=mol_data, pro_data=pro_data)
        test_dataset = DTADataset(test_smile, test_seq, test_label, mol_data=mol_data, pro_data=pro_data)

    elif dataset  == 'PDBbind':
        train_id, train_label = list(df_train['id']), list(df_train['affinity'])
        test_id, test_label = list(df_test['id']), list(df_test['affinity'])
        train_dataset = PDBDataset(train_id, train_label, mol_data=mol_data, pro_data=pro_data)
        test_dataset = PDBDataset(test_id, test_label, mol_data=mol_data, pro_data=pro_data)

    train_loader = DrugDataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=args.num_workers)
    test_loader = DrugDataLoader(test_dataset, batch_size=args.batch, shuffle=False, num_workers=args.num_workers)


    # training the model
    model = MolDVC_DTA(vocab_protein_size = 26,embedding_size=128,out_dim=1,Alpha=args.Alpha,Beta=args.Beta,n_iter=args.n_iter).to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.LR)
    best_mse = 1000
    best_ci = 0
    best_epoch = -1
    model_file_name = f'results/{dataset}/'  + f'train_{model_st}_{dataset} + {args.window}.model'
    result_file_name = f'results/{dataset}/' + f'train_{model_st}_{dataset} + {args.window}.csv'
    for epoch in range(args.epochs):
        train(model, device, train_loader, optimizer, loss_fn, args, epoch + 1)
        G, P = test(model, device, test_loader)
        ret = [mse(G, P), rmse(G, P), mae(G,P), concordance_index(G, P),rm2(G, P)]
        if ret[0] < best_mse:
            torch.save(model.state_dict(), model_file_name)
            with open(result_file_name, 'w') as f:
                f.write(','.join(map(str, ret)))
            best_epoch = epoch + 1
            best_mse = ret[0]
            best_rmse = ret[1]
            best_mae = ret[2]
            best_ci = ret[3]
            best_rm2 = ret[4]
            print('rmse improved at epoch ', best_epoch, '; best_mse,best_rmse,best_mae,best_ci,best_rm2:', best_mse, best_rmse, best_mae,best_ci,best_rm2,dataset,model_st)
        else:
            print(ret[0], 'No improvement since epoch ', best_epoch, '; best_mse,best_rmse,best_mae,best_ci,best_rm2:', best_mse, best_rmse, best_mae,best_ci,best_rm2,dataset,model_st)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Alpha', type = float, default = 0.01, help='Coefficient of NG_Loss')
    parser.add_argument('--Beta',  type = float, default = 0.01, help='Coefficient of EG_Loss')
    parser.add_argument('--n_iter',type = int, default = 4, help='layers of sub_structure')
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--batch', type = int, default = 512)
    parser.add_argument('--LR', type = float, default = 0.0005)
    parser.add_argument('--log_interval', type = int, default = 20)
    parser.add_argument('--device', type = int, default = 1)
    parser.add_argument('--dataset', type = str, default = 'kiba',choices = ['davis','kiba','PDBbind'])
    parser.add_argument('--num_workers', type= int, default = 8)
    parser.add_argument('--window', type= int, default = 3)
    args = parser.parse_args()
    main(args)







