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
        if dataset_name in ["lipo", "freesolv", "esol"]:
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
        # Plot stacked bars with negative on top
        plt.figure(figsize=(7, 5))

        plt.rcParams.update({
            "axes.titlesize": 20,
            "axes.labelsize": 20,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16
        })

        if len(plot_df) == 1:
            # Center the single bar at x=0.5 with a width of 0.5
            x = [0.5]
            width = 0.5
            plt.bar(x, plot_df['positive'], width=width, label='Positive', color='lightblue')
            plt.bar(x, plot_df['negative'], width=width, bottom=plot_df['positive'], label='Negative', color='darkblue')
            plt.xticks([])  # No ticks needed for single bar
            plt.xlim(0, 1)  # Set x-axis limits to center the bar
        else:
            plt.bar(plot_df.index, plot_df['positive'], label='Positive', color='lightblue')
            plt.bar(plot_df.index, plot_df['negative'], bottom=plot_df['positive'], label='Negative', color='darkblue')
            plt.xticks([])
        plt.ylabel('Label Ratio')
        plt.xlabel('Task')
        #plt.title(f'Label Distribution for {dataset_name}')

        # handles, labels = plt.gca().get_legend_handles_labels()
        # plt.legend(handles[::-1], labels[::-1])

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
        output_dir = "dataset_statistics/figures/label_distribution"
        os.makedirs(output_dir, exist_ok=True)

        plt.rcParams.update({
            "axes.titlesize": 20,
            "axes.labelsize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16
        })

        # Create a histogram for these labels
        plt.figure(figsize=(7, 6))
        sns.histplot(labels, bins=30, kde=True)
        #plt.title(f"Label Distribution for {dataset_name}")
        plt.xlabel("Label Value")
        plt.ylabel("Frequency")
        plt.grid()

        # Save the figure
        output_path = f"{output_dir}/{dataset_name}_label_distribution.png"
        plt.savefig(output_path)
        plt.show()

    def visualize_all_labels_from_png(self):
            path_to_pngs = "dataset_statistics/figures/label_distribution"

            png_files = [f for f in os.listdir(path_to_pngs) if f.endswith('.png')]

            # Create a grid of subplots with 3 columns
            num_files = len(png_files)
            num_cols = 3
            num_rows = (num_files + num_cols - 1) // num_cols
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
            axes = axes.flatten() if num_rows > 1 else [axes]
            for ax, png_file in zip(axes, png_files):
                img = plt.imread(os.path.join(path_to_pngs, png_file))
                ax.imshow(img)
                ax.axis('off')
                ax.set_title(png_file.replace('_label_distribution.png', ''))
            # Hide any unused subplots
            for ax in axes[num_files:]:
                ax.axis('off')
            plt.tight_layout()

            # Save the combined figure
            save_path = "dataset_statistics/figures/all_label_distributions.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.show()