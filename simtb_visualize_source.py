import argparse
import math
import scipy
import matplotlib.pyplot as plt



# Visualize simtb source maps

# Usage: python simtb_visualize_source.py -s SOURCE

#   -s, --source : simtb source maps (.mat file)

if __name__ == '__main__':

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', type=str, required=True, help='simtb source maps (.mat file)')

    # parse and print command line arguments
    args = parser.parse_args()
    print('Source path:', args.source)
    print()

    ###################################################################################################################

    # load simtb data 
    data = scipy.io.loadmat(args.source)
    source_maps = data['SM']
    time_courses = data['TC']
    
    # display simtb source maps using matplotlib
    reshaped_source_maps = source_maps.reshape(source_maps.shape[0], math.isqrt(source_maps.shape[1]), math.isqrt(source_maps.shape[1]))
    fig, axes = plt.subplots(
        math.ceil(math.sqrt(source_maps.shape[0])), 
        math.ceil(math.sqrt(source_maps.shape[0])), 
        figsize=(10, 8))
    axes = axes.flatten()
    for i in range(source_maps.shape[0]):
        axes[i].imshow(reshaped_source_maps[i], cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(str(i + 1), fontsize=10, pad=2)
    plt.tight_layout()
    plt.show()
