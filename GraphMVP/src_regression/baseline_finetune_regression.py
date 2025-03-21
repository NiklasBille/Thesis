import argparse
import sys
from collections import OrderedDict

import numpy as np
import torch

sys.path.insert(0, '../src_classification')

from models_complete_feature import GNNComplete
from pretrain_JOAO import graphcl


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='GraphCL')
    parser.add_argument('--device', type=int, default=0, help='gpu')
    parser.add_argument('--batch_size', type=int, default=256, help='batch')
    parser.add_argument('--decay', type=float, default=0, help='weight decay')
    parser.add_argument('--epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--JK', type=str, default="last",
                        choices=['last', 'sum', 'max', 'concat'],
                        help='how the node features across layers are combined.')
    parser.add_argument('--gnn_type', type=str, default="gin", help='gnn model type')
    parser.add_argument('--dropout_ratio', type=float, default=0, help='dropout ratio')
    parser.add_argument('--emb_dim', type=int, default=300, help='embedding dimensions')
    parser.add_argument('--dataset', type=str, default=None, help='root dir of dataset')
    parser.add_argument('--num_layer', type=int, default=5, help='message passing layers')
    parser.add_argument('--output_model_file', type=str, default='', help='model save path')
    parser.add_argument('--num_workers', type=int, default=8, help='workers for dataset loading')

    parser.add_argument('--weight_path', type=str, default='')

    parser.add_argument('--aug_mode', type=str, default='sample')
    parser.add_argument('--aug_strength', type=float, default=0.2)

    # parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--output_model_dir', type=str, default='')
    args = parser.parse_args()
    print(args)

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) \
        if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    # set up model
    gnn = GNNComplete(num_layer=args.num_layer, emb_dim=args.emb_dim, JK=args.JK,
                      drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type)

    model = graphcl(gnn)

    model_state_dict = model.state_dict().keys()
    print(f"#layers for initialized model: {len(model_state_dict)}")
    print(f"state_dict for initialized model: {model_state_dict}\n")

    loaded_state_dict = torch.load(args.weight_path, map_location='cpu')
    # Add 'gnn.' in front of every key
    loaded_state_dict = {f'gnn.{k}': v for k, v in loaded_state_dict.items()}
    print(f"#layers for loaded weights: {len(loaded_state_dict.keys())}")
    print(f"state_dict for loaded weights: {loaded_state_dict.keys()}\n")

    common_layers = [layer for layer in model_state_dict if layer in loaded_state_dict]
    print(f"#layers in common: {len(common_layers)}")
    print(f"Layers in common: {common_layers}")


    #missing = ["gnn.atom_encoder.atom_embedding_list.0.weight", "gnn.atom_encoder.atom_embedding_list.1.weight", "gnn.atom_encoder.atom_embedding_list.2.weight", "gnn.atom_encoder.atom_embedding_list.3.weight", "gnn.atom_encoder.atom_embedding_list.4.weight", "gnn.atom_encoder.atom_embedding_list.5.weight", "gnn.atom_encoder.atom_embedding_list.6.weight", "gnn.atom_encoder.atom_embedding_list.7.weight", "gnn.atom_encoder.atom_embedding_list.8.weight", "gnn.gnns.0.eps", "gnn.gnns.0.mlp.0.weight", "gnn.gnns.0.mlp.0.bias", "gnn.gnns.0.mlp.1.weight", "gnn.gnns.0.mlp.1.bias", "gnn.gnns.0.mlp.1.running_mean", "gnn.gnns.0.mlp.1.running_var", "gnn.gnns.0.mlp.3.weight", "gnn.gnns.0.mlp.3.bias", "gnn.gnns.0.bond_encoder.bond_embedding_list.0.weight", "gnn.gnns.0.bond_encoder.bond_embedding_list.1.weight", "gnn.gnns.0.bond_encoder.bond_embedding_list.2.weight", "gnn.gnns.1.eps", "gnn.gnns.1.mlp.0.weight", "gnn.gnns.1.mlp.0.bias", "gnn.gnns.1.mlp.1.weight", "gnn.gnns.1.mlp.1.bias", "gnn.gnns.1.mlp.1.running_mean", "gnn.gnns.1.mlp.1.running_var", "gnn.gnns.1.mlp.3.weight", "gnn.gnns.1.mlp.3.bias", "gnn.gnns.1.bond_encoder.bond_embedding_list.0.weight", "gnn.gnns.1.bond_encoder.bond_embedding_list.1.weight", "gnn.gnns.1.bond_encoder.bond_embedding_list.2.weight", "gnn.gnns.2.eps", "gnn.gnns.2.mlp.0.weight", "gnn.gnns.2.mlp.0.bias", "gnn.gnns.2.mlp.1.weight", "gnn.gnns.2.mlp.1.bias", "gnn.gnns.2.mlp.1.running_mean", "gnn.gnns.2.mlp.1.running_var", "gnn.gnns.2.mlp.3.weight", "gnn.gnns.2.mlp.3.bias", "gnn.gnns.2.bond_encoder.bond_embedding_list.0.weight", "gnn.gnns.2.bond_encoder.bond_embedding_list.1.weight", "gnn.gnns.2.bond_encoder.bond_embedding_list.2.weight", "gnn.gnns.3.eps", "gnn.gnns.3.mlp.0.weight", "gnn.gnns.3.mlp.0.bias", "gnn.gnns.3.mlp.1.weight", "gnn.gnns.3.mlp.1.bias", "gnn.gnns.3.mlp.1.running_mean", "gnn.gnns.3.mlp.1.running_var", "gnn.gnns.3.mlp.3.weight", "gnn.gnns.3.mlp.3.bias", "gnn.gnns.3.bond_encoder.bond_embedding_list.0.weight", "gnn.gnns.3.bond_encoder.bond_embedding_list.1.weight", "gnn.gnns.3.bond_encoder.bond_embedding_list.2.weight", "gnn.gnns.4.eps", "gnn.gnns.4.mlp.0.weight", "gnn.gnns.4.mlp.0.bias", "gnn.gnns.4.mlp.1.weight", "gnn.gnns.4.mlp.1.bias", "gnn.gnns.4.mlp.1.running_mean", "gnn.gnns.4.mlp.1.running_var", "gnn.gnns.4.mlp.3.weight", "gnn.gnns.4.mlp.3.bias", "gnn.gnns.4.bond_encoder.bond_embedding_list.0.weight", "gnn.gnns.4.bond_encoder.bond_embedding_list.1.weight", "gnn.gnns.4.bond_encoder.bond_embedding_list.2.weight", "gnn.batch_norms.0.weight", "gnn.batch_norms.0.bias", "gnn.batch_norms.0.running_mean", "gnn.batch_norms.0.running_var", "gnn.batch_norms.1.weight", "gnn.batch_norms.1.bias", "gnn.batch_norms.1.running_mean", "gnn.batch_norms.1.running_var", "gnn.batch_norms.2.weight", "gnn.batch_norms.2.bias", "gnn.batch_norms.2.running_mean", "gnn.batch_norms.2.running_var", "gnn.batch_norms.3.weight", "gnn.batch_norms.3.bias", "gnn.batch_norms.3.running_mean", "gnn.batch_norms.3.running_var", "gnn.batch_norms.4.weight", "gnn.batch_norms.4.bias", "gnn.batch_norms.4.running_mean", "gnn.batch_norms.4.running_var", "projection_head.0.weight", "projection_head.0.bias", "projection_head.2.weight", "projection_head.2.bias"]
    #unexpected = ["x_embedding1.weight", "x_embedding2.weight", "gnns.0.mlp.0.weight", "gnns.0.mlp.0.bias", "gnns.0.mlp.2.weight", "gnns.0.mlp.2.bias", "gnns.0.edge_embedding1.weight", "gnns.0.edge_embedding2.weight", "gnns.1.mlp.0.weight", "gnns.1.mlp.0.bias", "gnns.1.mlp.2.weight", "gnns.1.mlp.2.bias", "gnns.1.edge_embedding1.weight", "gnns.1.edge_embedding2.weight", "gnns.2.mlp.0.weight", "gnns.2.mlp.0.bias", "gnns.2.mlp.2.weight", "gnns.2.mlp.2.bias", "gnns.2.edge_embedding1.weight", "gnns.2.edge_embedding2.weight", "gnns.3.mlp.0.weight", "gnns.3.mlp.0.bias", "gnns.3.mlp.2.weight", "gnns.3.mlp.2.bias", "gnns.3.edge_embedding1.weight", "gnns.3.edge_embedding2.weight", "gnns.4.mlp.0.weight", "gnns.4.mlp.0.bias", "gnns.4.mlp.2.weight", "gnns.4.mlp.2.bias", "gnns.4.edge_embedding1.weight", "gnns.4.edge_embedding2.weight", "batch_norms.0.weight", "batch_norms.0.bias", "batch_norms.0.running_mean", "batch_norms.0.running_var", "batch_norms.0.num_batches_tracked", "batch_norms.1.weight", "batch_norms.1.bias", "batch_norms.1.running_mean", "batch_norms.1.running_var", "batch_norms.1.num_batches_tracked", "batch_norms.2.weight", "batch_norms.2.bias", "batch_norms.2.running_mean", "batch_norms.2.running_var", "batch_norms.2.num_batches_tracked", "batch_norms.3.weight", "batch_norms.3.bias", "batch_norms.3.running_mean", "batch_norms.3.running_var", "batch_norms.3.num_batches_tracked", "batch_norms.4.weight", "batch_norms.4.bias", "batch_norms.4.running_mean", "batch_norms.4.running_var", "batch_norms.4.num_batches_tracked"]
    #print(len(missing))
    #print(len(unexpected))
    #print(model)
    #model.to(device)
