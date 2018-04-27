import numpy as np
import scipy as sp
import scipy.ndimage as ndimage
import cv2
from matplotlib import pyplot as plt

def polyfit_to_components(bw_img, degree=1):
    """
    Description: Fits a ployline to the connected components in the binary image
    :param bw_img: binary image
    :param degree: degree of polynomial
    :return: polynomial coefficients {c0, c1, ..., cn}, y = c0 + c1*x + ...
    """
    bw = (bw_img)> 0.1
    if (np.sum(bw) > 0):
        [row, col] = np.where(bw > 0)
        sorted_col = np.sort(col)
        Y = np.zeros(sorted_col.shape, dtype='float')
        for ii in range(len(Y)):
            X_i = sorted_col(ii)                                                                                        # Represent data as point cloud {(xi, yi)}
            y_for_xi = np.where(bw[:, X_i] == 1)                                                                        # All points corresponding to xi
            Y[ii] = np.mean(Y_for_xi)

    A = np.zeros((len(Y), degree+1), dtype='float')
    for jj in range(degree+1):
        A[:, jj] = sorted_col ** jj                                                                                     # A[:,k]=x^k,
        B = np.linalg.lstsq(A, Y, rcond=-1)

    return B[0]