import torch
import torch.nn as nn
import torch.nn.functional as F
#from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.nn import (MessagePassing, global_add_pool,
                                global_max_pool, global_mean_pool)
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, softmax
from torch_scatter import scatter_add


from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims 

class AtomEncoder(torch.nn.Module):
    def __init__(self, emb_dim, optional_full_atom_features_dims=None):
        super(AtomEncoder, self).__init__()


        self.atom_embedding_list = torch.nn.ModuleList()

        if optional_full_atom_features_dims is not None:
            full_atom_feature_dims = optional_full_atom_features_dims
        else:
            full_atom_feature_dims = get_atom_feature_dims()

        for i, dim in enumerate(full_atom_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:,i])

        return x_embedding


class BondEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super(BondEncoder, self).__init__()

        full_bond_feature_dims = get_bond_feature_dims()

        self.bond_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_bond_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:,i])

        return bond_embedding   


class GINConv(MessagePassing):
    def __init__(self, emb_dim, aggr="add"):
        super(GINConv, self).__init__(aggr=aggr)

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))
        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class GCNConv(MessagePassing):
    def __init__(self, emb_dim, aggr="add"):
        super(GCNConv, self).__init__(aggr=aggr)

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.bond_encoder(edge_attr)

        row, col = edge_index

        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr = edge_embedding, norm=norm) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1,1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class GNNComplete(nn.Module):
    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0., gnn_type="gin"):

        if num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        super(GNNComplete, self).__init__()
        self.drop_ratio = drop_ratio
        self.num_layer = num_layer
        self.JK = JK

        self.atom_encoder = AtomEncoder(emb_dim)

        ###List of MLPs
        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr="add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim, aggr="add"))

        ###List of batchnorms
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

    # def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.atom_encoder(x)

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            # h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]
        else:
            raise ValueError("not implemented.")
        return node_representation


class GNN_graphpredComplete(nn.Module):
    def __init__(self, args, num_tasks, molecule_model=None):
        super(GNN_graphpredComplete, self).__init__()

        if args.num_layer < 2:
            raise ValueError("# layers must > 1.")

        self.molecule_model = molecule_model
        self.num_layer = args.num_layer
        self.emb_dim = args.emb_dim
        self.num_tasks = num_tasks
        self.JK = args.JK

        # Different kind of graph pooling
        if args.graph_pooling == "sum":
            self.pool = global_add_pool
        elif args.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif args.graph_pooling == "max":
            self.pool = global_max_pool
        else:
            raise ValueError("Invalid graph pooling type.")

        # For graph-level binary classification
        self.mult = 1

        if self.JK == "concat":
            self.graph_pred_linear = nn.Linear(self.mult * (self.num_layer + 1) * self.emb_dim,
                                               self.num_tasks)
        else:
            self.graph_pred_linear = nn.Linear(self.mult * self.emb_dim, self.num_tasks)
        return

    def from_pretrained(self, model_file):
        self.molecule_model.load_state_dict(torch.load(model_file))
        return

    def get_graph_representation(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, \
                                              data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.molecule_model(x, edge_index, edge_attr)
        graph_representation = self.pool(node_representation, batch)
        pred = self.graph_pred_linear(graph_representation)

        return graph_representation, pred

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, \
                                              data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.molecule_model(x, edge_index, edge_attr)
        graph_representation = self.pool(node_representation, batch)
        output = self.graph_pred_linear(graph_representation)

        return output
