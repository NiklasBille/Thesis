import RegressionLabelPlotter as rlp
import FeatureCounter as fc

if __name__ == '__main__':
    
    # Visualize the label distribution
    # plotter = rlp.RegressionLabelPlotter()
    # plotter.visualize_label_distribution('lipo')
    # plotter.visualize_label_distribution('freesolv')
    # plotter.visualize_label_distribution('esol')

    feature_counter  = fc.FeatureCounter()
    feature_counter.print_features('hiv')



    
    