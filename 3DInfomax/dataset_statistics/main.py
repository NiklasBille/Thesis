import LabelPlotter as lp
import FeatureCounter as fc

if __name__ == '__main__':
    
    # Visualize the label distribution
    datasets = ['lipo', 'freesolv', 'esol', 'tox21', 'toxcast', 'clintox', 'bace', 'bbbp', 'hiv', 'sider', 'muv']

    plotter = lp.LabelPlotter()
    plotter.visualize_label_distribution("bace")
    feature_counter = fc.FeatureCounter()

    # dataset_statistics = {}
    # for dataset in datasets:
    #     avg_atom_count = feature_counter.get_average_atom_count(dataset)
    #     avg_bond_count_per_atom = feature_counter.get_average_bond_count_per_atom(dataset)
    #     dataset_statistics[dataset] = {
    #         'avg_atom_count': avg_atom_count,
    #         'avg_bond_count_per_atom': avg_bond_count_per_atom
    #     }
    
    # # Print the dataset statistics
    # for dataset, stats in dataset_statistics.items():
    #     print(f"Dataset: {dataset}")
    #     print(f"Average Atom Count: {stats['avg_atom_count']}")
    #     print(f"Average Bond Count per Atom: {stats['avg_bond_count_per_atom']}")
    #     print("-" * 30)

    



    
    