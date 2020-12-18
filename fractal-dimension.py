# -----------------------------------------------------------------------------
# From https://en.wikipedia.org/wiki/Minkowski–Bouligand_dimension:
#
# In fractal geometry, the Minkowski–Bouligand dimension, also known as
# Minkowski dimension or box-counting dimension, is a way of determining the
# fractal dimension of a set S in a Euclidean space Rn, or more generally in a
# metric space (X, d).
# -----------------------------------------------------------------------------
import scipy.misc
import numpy as np
import imageio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def fractal_dimension(Z, threshold=0.5):

    # Only for 2d image
    assert(len(Z.shape) == 2)

    # From https://github.com/rougier/numpy-100 (#87)
    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                               np.arange(0, Z.shape[1], k), axis=1)

        # We count non-empty (0) and non-full boxes (k*k)
        return len(np.where((S > 0) & (S < k*k))[0])


    # Transform Z into a binary array
    Z = (Z < threshold)
    imgplot = plt.imshow(Z)
    plt.colorbar()
    plt.savefig('leaf%d.png'%(len(Z)))
    plt.clf()
    # Minimal dimension of image
    p = min(Z.shape)

    # Greatest power of 2 less than or equal to p
    print(p)
    n = 2**np.floor(np.log(p)/np.log(2))

    # Extract the exponent
    n = int(np.log(n)/np.log(2))

    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2**np.arange(n, 1, -1)

    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))
    print(counts)
    print(sizes)
    # Fit the successive log(sizes) with log (counts)
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    plt.plot(np.log(sizes),np.log(counts), 'o', mfc='none')
    plt.plot(np.log(sizes), np.polyval(coeffs,np.log(sizes)))
    plt.xlabel('log $\epsilon$')
    plt.ylabel('log N')
    plt.savefig('sierpinski_dimension%d.png'%(len(Z)))
    plt.clf()
    return -coeffs[0]

I1 =  rgb2gray(imageio.imread("fern.png"))/255
I2 =  rgb2gray(imageio.imread("leaf.jpg"))/255
print("Minkowski–Bouligand dimenion of fern(computed): ", fractal_dimension(I1, threshold=0.5))
print("Minkowski–Bouligand dimenion of leaf(computed): ", fractal_dimension(I2, threshold=0.8))


