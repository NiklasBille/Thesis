import LabelPlotter as lp
import FeatureCounter as fc

if __name__ == '__main__':
    
    # Visualize the label distribution
    datasets = ["lipo", "freesolv", "esol", "tox21", "toxcast", "clintox", "bace", "bbbp", "hiv", "sider", "muv"]

    plotter = lp.LabelPlotter()
    for dataset in datasets:
        print(f"Visualizing label distribution for dataset: {dataset}")
        plotter.visualize_label_distribution(dataset)
    # plotter.visualize_all_labels_from_png()
    