from ogb.utils import features
import sys
import torch
def _sample_uniform_categorical(num: int, size: int) -> torch.Tensor:
    logits = torch.log(torch.ones(size) / size).repeat(num, 1)
    return torch.distributions.Categorical(logits=logits).sample()


from ogb.utils import features

def get_noisy_atom_features(
    atom_features_tensor: torch.Tensor,
    sample_random: float = 0.05
) -> torch.Tensor:
    """
    Adds noise to an existing atom feature tensor by randomly replacing values.

    Args:
        atom_features_tensor: Tensor of shape [num_atoms, feature_dim], dtype long.
        sample_random: Probability of replacing a feature with a random valid value.

    Returns:
        Tensor of same shape with some noisy values.
    """
    num_atoms, feat_dim = atom_features_tensor.shape
    noisy_features = []

    for i in range(feat_dim):
        # Finds all possible noise values for given feature
        vocab_size = features.get_atom_feature_dims()[i]
        # Samples a random value for each atom 
        sampled_column = _sample_uniform_categorical(num_atoms, vocab_size)
        # Create a boolean mask for which atoms to add noise to
        noise_mask = torch.rand(num_atoms) < sample_random
        # Adds noise
        noisy_col = torch.where(noise_mask, sampled_column, atom_features_tensor[:, i])
        noisy_features.append(noisy_col)
    

    return torch.stack(noisy_features, dim=1)

def get_noisy_edge_features(
    edge_features_tensor: torch.Tensor,
    sample_random: float = 0.05
) -> torch.Tensor:
    """
    Adds noise to an existing edge feature tensor by randomly replacing values.

    Args:
        edge_features_tensor: Tensor of shape [num_edges, feature_dim], dtype long.
        sample_random: Probability of replacing a feature with a random valid value.

    Returns:
        Tensor of same shape with some noisy values.
    """
    num_edges, feat_dim = edge_features_tensor.shape
    noisy_features = []

    for i in range(feat_dim):
        # Get the number of possible values for the current edge feature
        vocab_size = features.get_bond_feature_dims()[i]
        # Sample a random value for each edge
        sampled_column = _sample_uniform_categorical(num_edges, vocab_size)
        # Create a boolean mask for which edges to add noise to
        noise_mask = torch.rand(num_edges) < sample_random
        # Add noise
        noisy_col = torch.where(noise_mask, sampled_column, edge_features_tensor[:, i])
        noisy_features.append(noisy_col)

    return torch.stack(noisy_features, dim=1)

