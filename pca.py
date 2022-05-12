# Import Libraries
import utils
import numpy as np
from scipy.linalg import eigh
import sys

def PCA(X, components):
    X_mean = np.mean(X, axis=1)[:, np.newaxis]
    X_centered = X - X_mean
    corr = X_centered @ X_centered.T
    n = np.shape(corr)[0]
    s, V = eigh(corr, check_finite=False, driver="evx", subset_by_index=[n - components, n - 1])
    V = np.fliplr(V)
    return V.T @ X_centered

def main():
    X, Y = utils.load_data(sys.argv[1])
    X_np = X.to_numpy(dtype="float32").T
    Y_np = Y.to_numpy()
    X_reduced = PCA(X_np, 2)

    utils.scatter_plot(X_reduced, Y_np, 10)

if __name__ == "__main__":
    main()