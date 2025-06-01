import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets.ogbg_dataset_extension import OGBGDatasetExtension

class LabelPlotter:
    def __init__(self):
        pass        
        
    def visualize_label_distribution(self, dataset_name=None):
        if dataset_name in ["lipo", "freesolve", "esol"]:
            self._visualize_regression_label_distribution(dataset_name)
        else:
            self._visualize_classification_label_distribution(dataset_name)

    def _visualize_classification_label_distribution(self, dataset_name=None):
        dataset = OGBGDatasetExtension(name=f'ogbg-mol{dataset_name}', return_types=['targets'], device='cuda')

        # Each label is a tensor with multiple elements/tasks
        labels = [dataset[i][0].tolist() for i in range(len(dataset))]
        df_labels = pd.DataFrame(labels)

        # Count 0s and 1s for each task (column)
        label_counts = df_labels.apply(lambda x: x.value_counts(dropna=False), axis=0).fillna(0)

        negative_percentage = label_counts.loc[0] / (label_counts.loc[0] + label_counts.loc[1])
        positive_percentage = 1 - negative_percentage

        # Combine into a DataFrame for plotting
        plot_df = pd.DataFrame({
            'task': range(len(positive_percentage)),
            'positive': positive_percentage.values,
            'negative': negative_percentage.values
        })

        # Sort by positive percentage (descending)
        plot_df = plot_df.sort_values(by='positive', ascending=False).reset_index(drop=True)
        print(plot_df)
        # Plot stacked bars with negative on top
        plt.figure(figsize=(10, 6))
        plt.bar(plot_df.index, plot_df['positive'], label='Positive', color='lightblue')
        plt.bar(plot_df.index, plot_df['negative'], bottom=plot_df['positive'], label='Negative', color='darkblue')

        plt.xlabel('Task (sorted by positive ratio)')
        plt.ylabel('Label Ratio')
        plt.title(f'Label Distribution for ogbg-mol{dataset_name}')
        plt.legend()
        plt.xticks([])

        output_dir = "dataset_statistics/figures/label_distribution"
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/{dataset_name}_label_distribution.png"
        plt.savefig(output_path)

        plt.show()    


    def _visualize_regression_label_distribution(self, dataset_name=None):
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