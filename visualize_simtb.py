import sys
import scipy
import numpy as np
import matplotlib.pyplot as plt



# Usage: python visualize_simtb.py <SIM FILENAME>.mat

if __name__ == '__main__':

    # load simtb data 
    if len(sys.argv) != 2:
        raise Exception('Usage: python visualize_simtb.py <SIM FILENAME>.mat')
    data_path = sys.argv[1]
    data = scipy.io.loadmat(data_path)
    source_maps = data['SM']
    time_courses = data['TC']
    
    # display simtb source maps using matplotlib
    reshaped_source_maps = source_maps.reshape(20, 128, 128)
    fig, axes = plt.subplots(4, 5, figsize=(10, 8))
    axes = axes.flatten()
    for i in range(20):
        axes[i].imshow(reshaped_source_maps[i], cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(str(i + 1), fontsize=10, pad=2)
    plt.tight_layout()
    plt.show()

    # calculate time course correlation and print to console
    tc_correlation = np.zeros((20, 20))
    for i in range(20):
        for j in range(20):
            r, _ = scipy.stats.pearsonr(time_courses[:, i], time_courses[:, j])
            tc_correlation[i, j] = r
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=2)
    np.set_printoptions(linewidth=100000)
    print(tc_correlation)