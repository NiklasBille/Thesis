import copy
import os
from os.path import join
import sys
from sklearn.linear_model import LogisticRegression # without this import before the rest we deadlock (very funky) 

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from datetime import datetime 
from config import args
from models import GNN, GNN_graphpred
from sklearn.metrics import (accuracy_score, average_precision_score,
                             roc_auc_score)
from splitters import random_scaffold_split, random_split, scaffold_split
from torch_geometric.data import DataLoader
from util import get_num_task
from torch.utils.tensorboard import SummaryWriter
import pyaml
from datasets import MoleculeDataset


def train(model, device, loader, optimizer):
    model.train()
    total_loss = 0

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        # Whether y is non-null or not.
        is_valid = y ** 2 > 0
        # Loss matrix
        loss_mat = criterion(pred.double(), (y + 1) / 2)
        # loss matrix after removing null target
        loss_mat = torch.where(
            is_valid, loss_mat,
            torch.zeros(loss_mat.shape).to(device).to(loss_mat.dtype))

        optimizer.zero_grad()
        loss = torch.sum(loss_mat) / torch.sum(is_valid)
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item()

    return total_loss / len(loader)


def eval(model, device, loader, compute_loss=False):
    """
    Evaluate a model on a dataset and compute ROC-AUC and PRC-AUC.

    Args:
        model: The PyTorch model to evaluate.
        device: Device to run the model on.
        loader: DataLoader for the evaluation dataset.
        compute_loss (bool): Whether to compute and return average loss. 

    Returns:
        metrics (dict): ROC-AUC and PRC-AUC scores.
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        avg_loss (float, optional): Average loss if compute_loss is True.
    """
    model.eval()
    y_true, y_scores = [], []

    total_loss = 0
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    
        true = batch.y.view(pred.shape)

        y_true.append(true)
        y_scores.append(pred)

        if compute_loss:
            # Whether y is non-null or not.
            is_valid = true ** 2 > 0
            # Loss matrix
            loss_mat = criterion(pred.double(), (true + 1) / 2)
            # loss matrix after removing null target
            loss_mat = torch.where(
                is_valid, loss_mat,
                torch.zeros(loss_mat.shape).to(device).to(loss_mat.dtype))
            
            loss = torch.sum(loss_mat) / torch.sum(is_valid)
            total_loss += loss.detach().item()

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    roc_list, prc_list = calculate_roc_and_prc(y_true, y_scores)

    if len(roc_list) < y_true.shape[1]:
        print(len(roc_list))
        print('Some target is missing!')
        print('Missing ratio: %f' %(1 - float(len(roc_list)) / y_true.shape[1]))

    if compute_loss:
        return {'ROC': sum(roc_list) / len(roc_list), 'PRC': sum(prc_list) / len(prc_list)}, y_true, y_scores, total_loss/len(loader)
    else:
        return {'ROC': sum(roc_list) / len(roc_list), 'PRC': sum(prc_list) / len(prc_list)}, y_true, y_scores

def calculate_roc_and_prc(y_true, y_scores):
    roc_list = []
    prc_list = []
    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == -1) > 0:
            is_valid = y_true[:, i] ** 2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid, i] + 1) / 2, y_scores[is_valid, i]))
            prc_list.append(average_precision_score((y_true[is_valid, i] + 1) / 2, y_scores[is_valid, i]))
        else:
            print('{} is invalid'.format(i))
    return roc_list, prc_list

def log_scalars_to_tensorboard(writer, results, losses):
    """
    Log ROC-AUC, PRC-AUC, and loss to TensorBoard for each data split.

    Args:
        writers (dict): TensorBoard SummaryWriter objects for per split.
        results (dict): Dictionary with ROC-AUC and PRC-AUC metrics per split.
        losses (dict): Dictionary with loss values per split.
    """
    for split in ['train', 'val', 'test']:
        result = results[split]
        loss = losses[split]
        writer.add_scalar(f'rocauc/{split}', result['ROC'], epoch)
        writer.add_scalar(f'prcauc/{split}', result['PRC'], epoch)
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

    split_naming = 'scaff' if args.split == 'scaffold' else 'random' # for directory naming reasons
    log_dir = f"{args.output_model_dir}/{args.dataset}_{split_naming}_{args.runseed}_{datetime.now().strftime('%d-%m_%H-%M-%S')}"
    print("[ Logs to : ", log_dir, " ]")

    seed_all(args.runseed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')
    print("[ Using device : ", device, " ]")

    # create writers for Tensorboard
    writer = SummaryWriter(log_dir)

    # Bunch of classification tasks
    num_tasks = get_num_task(args.dataset)
    dataset_folder = '../datasets/molecule_datasets/'
    dataset = MoleculeDataset(dataset_folder + args.dataset, dataset=args.dataset)
    if args.noise_level > 0:
        dataset_noise = MoleculeDataset(dataset_folder + args.dataset, dataset=args.dataset, noise_level=args.noise_level)
    print(dataset)

    remaining_prop = (1-args.train_prop)/2

    if args.split == 'scaffold':
        smiles_list = pd.read_csv(dataset_folder + args.dataset + '/processed/smiles.csv',
                                  header=None)[0].tolist()
        # We only add noise to the train_dataset  
        train_dataset, valid_dataset, test_dataset = scaffold_split(
            dataset, smiles_list, null_value=0, frac_train=args.train_prop,
            frac_valid=remaining_prop, frac_test=remaining_prop)
        
        if args.noise_level > 0:
            # Overwrite the train_dataset with noise
            train_dataset, _, _ = scaffold_split(
            dataset_noise, smiles_list, null_value=0, frac_train=args.train_prop,
            frac_valid=remaining_prop, frac_test=remaining_prop)
        print('split via scaffold')
    elif args.split == 'random':
        train_dataset, valid_dataset, test_dataset = random_split(
            dataset, null_value=0, frac_train=args.train_prop, frac_valid=remaining_prop,
            frac_test=remaining_prop, seed=args.seed)
        print('randomly split')
    elif args.split == 'random_scaffold':
        smiles_list = pd.read_csv(dataset_folder + args.dataset + '/processed/smiles.csv',
                                  header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(
            dataset, smiles_list, null_value=0, frac_train=args.train_prop,
            frac_valid=remaining_prop, frac_test=remaining_prop, seed=args.seed)
        print('random scaffold')
    else:
        raise ValueError('Invalid split option.')
    print(train_dataset[0])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    if log_dir != '' and args.config is not None:
        train_args = copy.copy(args)
        config_path = args.config if isinstance(args.config, str) else args.config.name
        train_args.config = os.path.join(log_dir, os.path.basename(config_path))
        with open(os.path.join(log_dir, 'train_arguments.yaml'), 'w') as yaml_path:
            pyaml.dump(train_args.__dict__, yaml_path)
            
    # set up model
    molecule_model = GNN(num_layer=args.num_layer, emb_dim=args.emb_dim,
                         JK=args.JK, drop_ratio=args.dropout_ratio,
                         gnn_type=args.gnn_type)
    model = GNN_graphpred(args=args, num_tasks=num_tasks,
                          molecule_model=molecule_model)
    if not args.input_model_file == '':
        model.from_pretrained(args.input_model_file)
    model.to(device)
    print(model)

    # set up optimizer
    # different learning rates for different parts of GNN
    model_param_group = [{'params': model.molecule_model.parameters()},
                         {'params': model.graph_pred_linear.parameters(),
                          'lr': args.lr * args.lr_scale}]
    optimizer = optim.Adam(model_param_group, lr=args.lr,
                           weight_decay=args.decay)
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    
    train_result_list, val_result_list, test_result_list = [], [], []
    metric_list = ['ROC', 'PRC']
    best_val_roc, best_val_idx = -1, 0

    for epoch in range(1, args.epochs + 1):
        loss_acc = train(model, device, train_loader, optimizer)
        print('Epoch: {}\nLoss: {}'.format(epoch, loss_acc))

        if args.eval_train:
            train_result, train_target, train_pred = eval(model, device, train_loader)
        else:
            train_result = {'ROC': 0, 'PRC': 0}
    
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

        if val_result['ROC'] > best_val_roc:
            best_val_roc = val_result['ROC']
            best_val_idx = epoch - 1
            if not log_dir == '':
                output_model_path = join(log_dir, 'model_best.pth')
                saved_model_dict = {
                    'molecule_model': molecule_model.state_dict(),
                    'model': model.state_dict()
                }
                torch.save(saved_model_dict, output_model_path)

                filename = join(log_dir, 'evaluation_best.pth')
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
    with open(join(log_dir, 'evaluation_test.txt'), 'w') as f:
        f.write(f'prcauc: {test_result_list[best_val_idx]["PRC"]}\n'
                f'rocauc: {test_result_list[best_val_idx]["ROC"]} ')

    if log_dir is not '':
        output_model_path = join(log_dir, 'model_final.pth')
        saved_model_dict = {
            'molecule_model': molecule_model.state_dict(),
            'model': model.state_dict()
        }
        torch.save(saved_model_dict, output_model_path)
