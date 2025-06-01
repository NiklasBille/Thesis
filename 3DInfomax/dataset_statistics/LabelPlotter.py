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
        
        # for each task compute the ratio of positive and negative lalels
        label_counts = pd.DataFrame(labels).apply(lambda x: x.value_counts(), axis=0)
        # Compute the ratio of positive and negative labels without removing the NaN values. Act as the length of eask task are the positive and negative labels combined
        negative_percentage = label_counts.iloc[0, :] / (label_counts.iloc[0, :] + label_counts.iloc[1, :])
        # Sort the labels by the ratio of negative labels
        negative_percentage = negative_percentage.sort_values(ascending=False)
        positive_percentage = 1 - negative_percentage
        
        # Use the sns barplot to plot the ratio of positive and negative labels
        f, ax = plt.subplots(figsize=(10, 6))
        sns.set_theme(style="whitegrid")
        sns.set_color_codes("pastel")

        # Background bar (Negative)
        sns.barplot(x=negative_percentage, y=negative_percentage.index, label="Negative", color="b")
        # Overlay bar (Positive)
        sns.set_color_codes("muted")
        sns.barplot(x=positive_percentage, y=positive_percentage.index, label="Positive", color="b")
        # Save the figure
        output_dir = "dataset_statistics/figures/label_distribution"
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/{dataset_name}_label_distribution.png"
        plt.savefig(output_path)
        plt.show()

        


    def _visualize_clintox_label_distribution(self, dataset_name=None):

        # Load the dataset
        dataset = OGBGDatasetExtension(name=f'ogbg-mol{dataset_name}', return_types=['targets'], device='cuda')
        labels = [dataset[i][0].tolist() for i in range(len(dataset))]

        # Each label is a tensor with 2 elements/tasks
        labels_count_0 = pd.Series([label[0] for label in labels]).value_counts().sort_index()
        labels_count_1 = pd.Series([label[1] for label in labels]).value_counts().sort_index()

        # Define fixed colors: label 0 -> blue, label 1 -> orange
        label_colors = {0: '#1f77b4', 1: '#ff7f0e'}  # matplotlib default colors for consistency

        # Map colors in correct order
        colors_0 = [label_colors[label] for label in labels_count_0.index]
        colors_1 = [label_colors[label] for label in labels_count_1.index]

        # Create the pie charts
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.pie(labels_count_0, labels=labels_count_0.index, autopct='%1.1f%%',
                startangle=140, colors=colors_0)
        plt.title(f"Label Distribution for {dataset_name} - Task 1")
        plt.axis('equal')
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.pie(labels_count_1, labels=labels_count_1.index, autopct='%1.1f%%',
                startangle=140, colors=colors_1)
        plt.title(f"Label Distribution for {dataset_name} - Task 2")
        plt.axis('equal')
        plt.grid()

        # Save the figure
        output_dir = "dataset_statistics/figures/label_distribution"
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/{dataset_name}_label_distribution.png"
        plt.savefig(output_path)
        plt.show()



        
    def _visualize_many_class_label_distribution(self, dataset_name=None):
        # Load the dataset
        dataset = OGBGDatasetExtension(name=f'ogbg-mol{dataset_name}', return_types=['targets'], device='cuda')
        # Each label is a tensor with 617 elements/tasks
        labels = [dataset[i][0].tolist() for i in range(len(dataset))]
        
        # for each task compute the ratio of positive and negative lalels
        label_counts = pd.DataFrame(labels).apply(lambda x: x.value_counts(), axis=0)
        # Compute the ratio of positive and negative labels without removing the NaN values. Act as the length of eask task are the positive and negative labels combined
        label_count_negative_percent = label_counts.iloc[0, :] / (label_counts.iloc[0, :] + label_counts.iloc[1, :])
        
        # Find the mean and standard deviation of the negative label counts
        mean_negative = label_count_negative_percent.mean()
        std_negative = label_count_negative_percent.std()
        
        # Make a pichart with the mean 
        plt.figure(figsize=(10, 6))
        # Use the mean_negative
        plt.pie([mean_negative, 1 - mean_negative], labels=['Mean negative label ratio', 'Mean positive label ratio'], autopct='%1.1f%%', startangle=140)
        # Create a legend with the standard deviation
        plt.title(f"Mean Label Distribution for {dataset_name} over {len(label_counts.columns)} tasks")
        plt.text(0.0, -1.2, f"Std of ratio over {len(label_counts.columns)} tasks : ±{std_negative:.2f}",
         ha='center', fontsize=10,
         bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.grid()
        # Save the figure
        output_dir = "dataset_statistics/figures/label_distribution"
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/{dataset_name}_label_distribution.png"
        plt.savefig(output_path)
        plt.show()       


def _visualize_single_class_label_distribution(self, dataset_name=None):
    # Load the dataset
    dataset = OGBGDatasetExtension(name=f'ogbg-mol{dataset_name}', return_types=['targets'], device='cuda')
    labels = [dataset[i][0].item() for i in range(len(dataset))]

    # Compute counts
    label_counts = pd.Series(labels).value_counts().sort_index()
    total = label_counts.sum()
    positive = label_counts.get(1, 0)
    negative = label_counts.get(0, 0)

    # Data for plotting
    plot_df = pd.DataFrame({
        'Label': ['All', 'All'],
        'Type': ['Negative', 'Positive'],
        'Ratio': [negative / total, positive / total]
    })

    # Initialize the matplotlib figure
    sns.set_theme(style="whitegrid")
    f, ax = plt.subplots(figsize=(6, 1.5))

    # Background bar (Negative)
    sns.set_color_codes("pastel")
    sns.barplot(x="Ratio", y="Label", data=plot_df[plot_df["Type"] == "Negative"],
                label="Negative", color="b")

    # Overlay bar (Positive)
    sns.set_color_codes("muted")
    sns.barplot(x="Ratio", y="Label", data=plot_df[plot_df["Type"] == "Positive"],
                label="Positive", color="b")

    # Add legend and axis labels
    ax.legend(ncol=2, loc="lower center", frameon=True, bbox_to_anchor=(0.5, -0.5))
    ax.set(xlim=(0, 1), ylabel="", xlabel="Proportion")
    sns.despine(left=True, bottom=True)

    # Save the figure
    output_dir = "dataset_statistics/figures/label_distribution"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{dataset_name}_label_distribution.png"
    plt.tight_layout()
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