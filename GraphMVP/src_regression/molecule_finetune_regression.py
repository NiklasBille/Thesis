import copy
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

sys.path.insert(0, '../src_classification')
from os.path import join

from config import args
from datasets_complete_feature import MoleculeDatasetComplete
from models_complete_feature import GNN_graphpredComplete, GNNComplete
from sklearn.metrics import mean_absolute_error, mean_squared_error
from splitters import random_scaffold_split, random_split, scaffold_split
from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pyaml

def train(model, device, loader, optimizer):
    model.train()
    total_loss = 0

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).squeeze()
        y = batch.y.squeeze()
        y = y.to(torch.float32) # float64 causes issues for some datasets, so convert to float32

        loss = reg_criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item()

    return total_loss / len(loader)


def eval(model, device, loader, compute_loss=False):
    """
    Evaluate a model on a dataset and compute RMSE and MAE.

    Args:
        model: The PyTorch model to evaluate.
        device: Device to run the model on.
        loader: DataLoader for the evaluation dataset.
        compute_loss (bool): Whether to compute and return average loss. 

    Returns:
        metrics (dict): RMSE and MAE scores.
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        avg_loss (float, optional): Average loss if compute_loss is True.
    """
    model.eval()
    y_true, y_pred = [], []
    
    total_loss = 0
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).squeeze(1)
 
        true = batch.y.view(pred.shape)
        y_true.append(true)
        y_pred.append(pred)

        if compute_loss:
            loss = reg_criterion(pred, true).detach().item()
            total_loss += loss

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_pred = torch.cat(y_pred, dim=0).cpu().numpy()
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    
    if compute_loss:
        return {'RMSE': rmse, 'MAE': mae}, y_true, y_pred, total_loss/len(loader)
    else:
        return {'RMSE': rmse, 'MAE': mae}, y_true, y_pred


def log_scalars_to_tensorboard(writer, results, losses):
    """
    Log MAE, RMSE, and loss to TensorBoard for each data split.

    Args:
        writers (dict): TensorBoard SummaryWriter objects for per split.
        results (dict): Dictionary with MAE and RMSE metrics per split.
        losses (dict): Dictionary with loss values per split.
    """
    for split in ['train', 'val', 'test']:
        result = results[split]
        loss = losses[split]
        writer.add_scalar(f'mae/{split}', result["MAE"], epoch)
        writer.add_scalar(f'rmse/{split}', result["RMSE"], epoch)
        writer.add_scalar(f'loss/{split}', loss, epoch)


def seed_all(seed):
    if not seed:
        seed = 0

    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # For PyTorch 1.8+. Required for running  torch.use_deterministic_algorithms()
    #torch.use_deterministic_algorithms(True)

if __name__ == '__main__':
    print('Arguments:')
    for k, v in vars(args).items():
        print(f'  {k}: {v}')

    seed_all(args.runseed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')

    print(device)
    sys.exit()


    # create writers for Tensorboard
    writer = SummaryWriter(args.output_model_dir)
    
    num_tasks = 1
    dataset_folder = '../datasets/molecule_datasets/'
    dataset_folder = os.path.join(dataset_folder, args.dataset)
    dataset = MoleculeDatasetComplete(dataset_folder, dataset=args.dataset)
    if args.noise_level > 0:
        dataset_noise = MoleculeDatasetComplete(dataset_folder, dataset=args.dataset, noise_level=args.noise_level)
    print('dataset_folder:', dataset_folder)
    print(dataset)

    if args.split == 'scaffold':
        smiles_list = pd.read_csv(dataset_folder + '/processed/smiles.csv', header=None)[0].tolist()
        # We only add noise to the train_dataset  
        train_dataset, valid_dataset, test_dataset = scaffold_split(
            dataset, smiles_list, null_value=0, frac_train=0.8,
            frac_valid=0.1, frac_test=0.1)
        
        if args.noise_level > 0:
            train_dataset, _, _ = scaffold_split(
                # Overwrite the train_dataset with noise
                dataset_noise, smiles_list, null_value=0, frac_train=0.8,
                frac_valid=0.1, frac_test=0.1)
        print('split via scaffold')
    elif args.split == 'random':
        train_dataset, valid_dataset, test_dataset = random_split(
            dataset, null_value=0, frac_train=0.8, frac_valid=0.1,
            frac_test=0.1, seed=args.seed)
        print('randomly split')
    elif args.split == 'random_scaffold':
        smiles_list = pd.read_csv(dataset_folder + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(
            dataset, smiles_list, null_value=0, frac_train=0.8,
            frac_valid=0.1, frac_test=0.1, seed=args.seed)
        print('random scaffold')
    else:
        raise ValueError('Invalid split option.')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    
    if args.output_model_dir != '' and args.config is not None:
        train_args = copy.copy(args)
        config_path = args.config if isinstance(args.config, str) else args.config.name
        train_args.config = os.path.join(args.output_model_dir, os.path.basename(config_path))
        with open(os.path.join(args.output_model_dir, 'train_arguments.yaml'), 'w') as yaml_path:
            pyaml.dump(train_args.__dict__, yaml_path)
    
    # set up model
    molecule_model = GNNComplete(num_layer=args.num_layer, emb_dim=args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type)
    model = GNN_graphpredComplete(args=args, num_tasks=num_tasks, molecule_model=molecule_model)
    if not args.input_model_file == '':
        model.from_pretrained(args.input_model_file)
    model.to(device)
    print(model)

    model_param_group = [
        {'params': model.molecule_model.parameters()},
        {'params': model.graph_pred_linear.parameters(), 'lr': args.lr * args.lr_scale}
    ]
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    # reg_criterion = torch.nn.L1Loss()
    reg_criterion = torch.nn.MSELoss()

    train_result_list, val_result_list, test_result_list = [], [], []
    # metric_list = ['RMSE', 'MAE', 'R2']
    metric_list = ['RMSE', 'MAE']
    best_val_rmse, best_val_idx = 1e10, 0

    for epoch in range(1, args.epochs + 1):
        loss_acc = train(model, device, train_loader, optimizer)
        print('Epoch: {}\nLoss: {}'.format(epoch, loss_acc))

        if args.eval_train:
            train_result, train_target, train_pred = eval(model, device, train_loader)
        else:
            train_result = {'RMSE': 0, 'MAE': 0, 'R2': 0}

        val_result, val_target, val_pred, val_loss = eval(model, device, val_loader, compute_loss=True)
        test_result, test_target, test_pred, test_loss = eval(model, device, test_loader, compute_loss=True)
        
        results = {'train': train_result, 'val': val_result, 'test': test_result}
        losses = {'train': loss_acc, 'val': val_loss, 'test': test_loss}
        log_scalars_to_tensorboard(writer, results, losses)


        train_result_list.append(train_result)
        val_result_list.append(val_result)
        test_result_list.append(test_result)

        for metric in metric_list:
            print('{} train: {:.6f}\tval: {:.6f}\ttest: {:.6f}'.format(metric, train_result[metric], val_result[metric], test_result[metric]))
        print()

        if val_result['RMSE'] < best_val_rmse:
            best_val_rmse = val_result['RMSE']
            best_val_idx = epoch - 1
            if not args.output_model_dir == '':
                output_model_path = join(args.output_model_dir, 'model_best.pth')
                saved_model_dict = {
                    'molecule_model': molecule_model.state_dict(),
                    'model': model.state_dict()
                }
                torch.save(saved_model_dict, output_model_path)

                filename = join(args.output_model_dir, 'evaluation_best.pth')
                np.savez(filename, val_target=val_target, val_pred=val_pred,
                         test_target=test_target, test_pred=test_pred)

    # Close Tensorboard writer
    writer.close()            
    
    # Print best validation metrics
    for metric in metric_list:
        print(f'Best ({metric}):\t train: {train_result_list[best_val_idx][metric]:.6f}\t'
                f'val: {val_result_list[best_val_idx][metric]:.6f}\t'
                f'test: {test_result_list[best_val_idx][metric]:.6f}')
    
    # Write best validation test metrics to a .txt file
    with open(join(args.output_model_dir, 'evaluation_test.txt'), 'w') as f:
        f.write(f'mae: {test_result_list[best_val_idx]["MAE"]}\n'
                f'rmse: {test_result_list[best_val_idx]["RMSE"]} ')
    
    if args.output_model_dir is not '':
        output_model_path = join(args.output_model_dir, 'model_final.pth')
        saved_model_dict = {
            'molecule_model': molecule_model.state_dict(),
            'model': model.state_dict()
        }
        torch.save(saved_model_dict, output_model_path)
