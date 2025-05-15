from collections import defaultdict
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tqdm import tqdm
from datasets.ogbg_dataset_extension import OGBGDatasetExtension
import pandas as pd
from rdkit import Chem
import pickle
class FeatureCounter:
    def __init__(self):
        pass
    
    def print_features(self, dataset_name=None):
        node_features_count, edge_features_count = self.count_features(dataset_name, save=True)
        # Print the node feature counts
        print("Node Feature Value Counts:")
        for feature_idx, value_counts in node_features_count.items():
            print(f"Feature {feature_idx}:")
            for val, count in sorted(value_counts.items()):
                print(f"  Value {val}: {count} times")
            
        # Print the edge feature counts
        print("\nEdge Feature Value Counts:")
        for feature_idx, value_counts in edge_features_count.items():
            print(f"Feature {feature_idx}:")
            for val, count in sorted(value_counts.items()):
                print(f"  Value {val}: {count} times")

    def visualize_features(self, dataset_name=None):
        node_features_count, edge_features_count = self.count_features(dataset_name, save=True)
        # Convert the counts to DataFrames for easier visualization
        node_df = pd.DataFrame.from_dict(node_features_count, orient='index').fillna(0)
        edge_df = pd.DataFrame.from_dict(edge_features_count, orient='index').fillna(0)

        # 

    def count_features(self, dataset_name=None, save=False):
        # If the dataset statistics have already been calculated, load them
        if os.path.exists(self.get_save_path(dataset_name, 'node')) and os.path.exists(self.get_save_path(dataset_name, 'edge')):
            node_features_dict = self.load_features(self.get_save_path(dataset_name, 'node'))
            edge_features_dict = self.load_features(self.get_save_path(dataset_name, 'edge'))
            return node_features_dict, edge_features_dict
        # Otherwise, calculate the statistics
        dataset = OGBGDatasetExtension(
            name=f'ogbg-mol{dataset_name}',
            return_types=['dgl_graph', 'target'],
            device='cuda'
        )
        # This tracks all unique values for each feature
        node_features_dict = defaultdict(set)
        edge_features_dict = defaultdict(set)

        # This counts how many times each value occurs for each feature
        node_features_count = defaultdict(dict)
        edge_features_count = defaultdict(dict)

        for n in tqdm(range(len(dataset))):
            graph, _ = dataset[n]
            node_features = graph.ndata['feat']  # (num_nodes, 9)
            edge_features = graph.edata['feat']  # (num_edges, 3)

            # Fix offset node feature 0 by 1 since molecule index starts from 0 instead of 1
            node_features[:, 0] += 1

            # Count node feature values
            for i in range(node_features.shape[0]):
                for j in range(node_features.shape[1]):
                    value = node_features[i][j].item()
                    # Count occurrence
                    if value in node_features_count[j]:
                        node_features_count[j][value] += 1
                    else:
                        node_features_count[j][value] = 1
                    # Track unique value
                    node_features_dict[j].add(value)

            # Count edge feature values
            for i in range(edge_features.shape[0]):
                for j in range(edge_features.shape[1]):
                    value = edge_features[i][j].item()
                    if value in edge_features_count[j]:
                        edge_features_count[j][value] += 1
                    else:
                        edge_features_count[j][value] = 1
                    edge_features_dict[j].add(value)

        # Optional: Sort values for readability
        node_features_dict = {k: sorted(v) for k, v in node_features_dict.items()}
        edge_features_dict = {k: sorted(v) for k, v in edge_features_dict.items()}

        # Save the unique values to a file
        if save:
            os.makedirs('dataset_statistics/feature_count', exist_ok=True)
            self.save_features(node_features_count, self.get_save_path(dataset_name, 'node'))
            self.save_features(edge_features_count, self.get_save_path(dataset_name, 'edge'))

        return node_features_count, edge_features_count
    
    def get_save_path(self, dataset_name, feature_type):
        if feature_type == 'edge':
            return f'dataset_statistics/feature_count/{dataset_name}_edge_features.pkl'
        elif feature_type == 'node':
            return f'dataset_statistics/feature_count/{dataset_name}_node_features.pkl'
        
    def save_features(self, obj, filename):
        with open(filename, 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

    def load_features(self, filename):
        with open(filename, 'rb') as inp:
            return pickle.load(inp)
    
