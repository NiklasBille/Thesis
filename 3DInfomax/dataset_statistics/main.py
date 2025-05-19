import LabelPlotter as lp
import FeatureCounter as fc

if __name__ == '__main__':
    
    # Visualize the label distribution
    datasets = ['lipo', 'freesolv', 'esol', 'tox21', 'toxcast', 'clintox', 'bace', 'bbbp', 'hiv', 'sider']

    plotter = lp.LabelPlotter()
    # plotter.visualize_label_distribution("sider")
    # plotter.visualize_label_distribution("tox21")

    plotter.visualize_label_distribution("clintox")
    plotter.visualize_label_distribution("toxcast")
    plotter.visualize_label_distribution("sider")
    plotter.visualize_label_distribution("tox21")


    



    
    