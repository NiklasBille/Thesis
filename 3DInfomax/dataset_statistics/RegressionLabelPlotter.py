import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets.ogbg_dataset_extension import OGBGDatasetExtension

class RegressionLabelPlotter:
    def __init__(self):
        pass
        
    def visualize_label_distribution(self, dataset_name=None):
        # Load the dataset
        dataset = OGBGDatasetExtension(name=f'ogbg-mol{dataset_name}', return_types=['targets'], device='cuda')
        labels = [dataset[i][0].item() for i in range(len(dataset))]

        # Ensure output directory exists
        output_dir = "dataset_statistics/figures"
        os.makedirs(output_dir, exist_ok=True)

        # Create a histogram for these labels
        plt.figure(figsize=(10, 6))
        sns.histplot(labels, bins=30, kde=True)
        plt.title(f"Label Distribution for {dataset_name}")
        plt.xlabel("Label Value")
        plt.ylabel("Frequency")
        plt.grid()

        # Save the figure
        output_path = f"{output_dir}/{dataset_name}_label_distribution.png"
        plt.savefig(output_path)
        plt.show()