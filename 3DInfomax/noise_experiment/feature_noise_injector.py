import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import torch
from typing import Dict, List
from noise_experiment.io import load_features
import random

class FeatureNoiseInjector:
    def __init__(self, 
                 dataset_name: str,
                 noise_probability: float,
                 device: torch.device = torch.device('cpu')):
        self.device = device
        node_features, edge_features = self._get_features_from_dataset(dataset_name)
        self.possible_node_values: Dict[int, List[int]]  = node_features
        self.possible_edge_values: Dict[int, List[int]]  = edge_features
        self.noise_probability = noise_probability
    
    def apply_noise(self, 
                    features_tensor: torch.Tensor,
                    feature_type: str = 'node'
                    ) -> torch.Tensor:
        """"
        Apply noise to a tensor of features
        """
        if feature_type == 'node':
            feature_values = self.possible_node_values
        elif feature_type == 'edge':
            feature_values = self.possible_edge_values
        else:
            raise ValueError("feature_type must be either 'node' or 'edge'.")
        return self._random_flip_noise(features_tensor, feature_values, self.noise_probability)      
    
        
    def _random_flip_noise(
            self,
            features_tensor: torch.Tensor,
            possible_values: Dict[int, List[int]],
            noise_probability: float
    ) -> torch.Tensor:
        """"
        Randomly flip features in the tensor with a given probability.
        """
        num_rows, feat_dim = features_tensor.shape
        noisy_features = []
        features_with_only_one_value = [i for i in range(feat_dim) if len(possible_values[i]) == 1]       
        # Loop over each feature dimension
        for j in range(feat_dim):
            # If the feature has only one possible value, we don't need to add noise
            if j in features_with_only_one_value:
                noisy_features.append(features_tensor[:, j])
                continue

            features_col = features_tensor[:, j]
            pos_values_for_feature = possible_values[j]
            noisy_features_col = features_col.clone()
            noise_mask = torch.rand(num_rows, device=self.device) < noise_probability

            # Get noisy indices
            noisy_indices = torch.nonzero(noise_mask, as_tuple=True)[0]
            if len(noisy_indices) > 0:
                # Get the original values at the noisy indices 
                values_to_be_flipped = features_col[noisy_indices]
                
                flipped_values = []
                for value in values_to_be_flipped:
                    choices = [v for v in pos_values_for_feature if v != value.item()]
                    flipped_values.append(random.choice(choices))
                
                noisy_features_col[noisy_indices] = torch.tensor(flipped_values, device=self.device)
            noisy_features.append(noisy_features_col)
        return torch.stack(noisy_features, dim=1)
    

    def _get_features_from_dataset(self, dataset_name: str):
        # Chech if its part of the ogbg datasets
        if 'ogbg' in dataset_name:
            node_features = load_features(f'noise_experiment/feature_values/{dataset_name}_node_features.pkl')
            edge_features = load_features(f'noise_experiment/feature_values/{dataset_name}_edge_features.pkl')
            return node_features, edge_features
        else:
            raise ValueError("Dataset not supported")

if __name__ == '__main__':
    # Create a tensor to test the noise injector
    test_edge_features = torch.tensor([[3, 2, 1], [1, 3, 1], [3, 2, 0]])

    # Create a noise injector
    noise_injector = FeatureNoiseInjector(
        node_feature_path='noise_experiment/feature_values/ogbg-molfreesolv_node_features.pkl',
        edge_feature_path='noise_experiment/feature_values/ogbg-molfreesolv_edge_features.pkl',
        noise_probability=0.1
    )

    # Apply noise to the tensor
    noisy_edge_features = noise_injector.apply_noise(test_edge_features, feature_type='edge')
