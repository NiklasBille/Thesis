import sys
import os 
import torch
from typing import Dict, List
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from feature_utils.extraction import load_features
import random
class FeatureNoiseInjector:
    def __init__(self, 
                 node_feature_path: str, 
                 edge_feature_path: str,
                 noise_probability: float,
                 device: torch.device = torch.device('cpu')):
        self.device = device
        self.possible_node_values: Dict[int, List[int]]  = load_features(node_feature_path)
        self.possible_edge_values: Dict[int, List[int]]  = load_features(edge_feature_path)
        self.noise_probability = noise_probability
    
    def apply_noise(self, 
                    features_tensor: torch.Tensor,
                    feature_type: str = 'node'
                    ) -> torch.Tensor:
        """"
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
        """
        Adds random flip noise to a categorical feature tensor.
        """
        num_rows, feat_dim = features_tensor.shape
        noisy_features = []
        for j in range(feat_dim):
            features_col = features_tensor[:, j]
            values = possible_values[j]
            noisy_features_col = features_col.clone()
            noise_mask = torch.rand(num_rows, device=self.device) < noise_probability

            for i in range(num_rows):
                if noise_mask[i]:
                    original_value = features_col[i].item()
                    # Checks if the original value is in the possible values
                    assert original_value in values, f"Original value {original_value} not in possible values {values}"
                    # Ensures edgecase where only one possible value exists
                    if len(values) == 1:
                        continue 
                    else: 
                        new_possible_values = values.copy()
                        new_possible_values.remove(original_value)
                        # replace original value with a random value from the possible values
                        noisy_features_col[i] = random.sample(new_possible_values, 1)[0]
            noisy_features.append(noisy_features_col)
        return torch.stack(noisy_features, dim=1)

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
