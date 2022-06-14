# -*- coding: utf-8 -*-
"""
@Author:    Ryan Lane
@Date:      05-12-2018
@Updated:   21-01-2022
"""

import numpy as np
from scipy.ndimage import convolve, uniform_filter
from skimage.morphology import disk
from skimage.filters import rank, sobel_h, sobel_v, laplace


# Default window size
WINDOW_SIZE = 15


def ACMO(image):
    """Absolute Central Moment

    Reference
    ---------
    [1] Shirvaikar (2004).
    """
    pass


def BREN(image):
    """Brenner's focus measure

    Reference
    ---------
    [2] Santos et al. (1997).
    """
    image = image.astype(np.int16)
    M, N = image.shape
    DH = np.zeros((M, N))
    DV = np.zeros((M, N))
    DH[:, :N - 2] = np.clip(image[:, 2:] - image[:, :-2], 0, None)
    DV[:M - 2, :] = np.clip(image[2:, :] - image[:-2, :], 0, None)
    FM = np.max((DH, DV), axis=0)**2
    return FM.mean()


def CURV(image):
    """Image curvature

    Reference
    ---------
    [3] Helmli & Scherer (2001).
    """
    pass


def GDER(image):
    """Image curvature

    Reference
    ---------
    [4] Geusebroek (2000).
    """
    N = np.floor(WINDOW_SIZE / 2)
    sig = N / 2.5
    y, x = np.mgrid[-N:N+1, -N:N+1]
    G = 1 / (2 * np.pi * sig) * np.exp(-(x ** 2 + y ** 2) / (2 * sig ** 2))
    Gx = -x * G/sig**2
    Gy = -y * G/sig**2
    Gx = Gx / np.sum(np.abs(Gx))
    Gy = Gy / np.sum(np.abs(Gy))
    Rx = convolve(image.astype(float), Gx, mode='nearest')
    Ry = convolve(image.astype(float), Gy, mode='nearest')
    FM = Rx**2 + Ry**2
    return FM.mean()


def GLVA(image):
    """Gray-level variance

    Reference
    ---------
    [5] Krotkov & Martin (1986).
    """
    FM = np.std(image, ddof=1)
    return FM


def GLLV(image):
    """Gray-level local variance

    Reference
    ---------
    [6] Pech-Pacheco et al. (2000).
    """
    def window_stdev(arr, r):
        c1 = uniform_filter(arr, r * 2, mode='mirror', origin=-r)
        c2 = uniform_filter(arr * arr, r * 2, mode='mirror', origin=-r)
        return ((c2 - c1*c1)**0.5)[:-r*2+1, :-r*2+1]
    LVar = window_stdev(image.astype(float), WINDOW_SIZE)**2
    FM = np.std(LVar, ddof=1)
    return FM


def GLVN(image):
    """Normalized gray-level variance

    Reference
    ---------
    [2] Santos et al. (1997).
    """
    FM = np.std(image, ddof=1)**2 / image.mean()
    return FM


def GRAE(image):
    """Energy of gradient

    Reference
    ---------
    [7] Subbarao et al. (1992).
    """
    image = image.astype(float)
    Ix = image.copy()
    Iy = image.copy()
    Ix[:, :-1] = np.clip(image[:, 1:] - image[:, :-1], a_min=0, a_max=None)
    Iy[:-1, :] = np.clip(image[1:, :] - image[:-1, :], a_min=0, a_max=None)
    FM = np.clip(Ix**2 + Iy**2, a_min=0, a_max=65535)
    return FM.mean()


def GRAT(image, thresh=0):
    """Thresholded gradient

    Reference
    ---------
    [2] Santos et al. (1997).
    """
    image = image.astype(float)
    Ix = image.copy()
    Iy = image.copy()
    Ix[:, :-1] = np.clip(image[:, 1:] - image[:, :-1], a_min=0, a_max=None)
    Iy[:-1, :] = np.clip(image[1:, :] - image[:-1, :], a_min=0, a_max=None)
    FM = np.max([np.abs(Ix), np.abs(Iy)], axis=0)
    FM[FM < thresh] = 0
    return FM.sum() / (FM!=0).sum()


def GRAS(image):
    """Squared gradient

    Reference
    ---------
    [8] Eskicioglu (1995).
    """
    Ix = np.clip(image[:, 1:] - image[:, :-1], a_min=0, a_max=None)
    FM = np.clip(Ix**2, a_min=0, a_max=65535)
    return FM.mean()


def HELM(image):
    """Helmli's mean method

    Reference
    ---------
    [3] Helmli & Scherer (2001).
    """
    selem = disk(WINDOW_SIZE)
    U = rank.mean(image, selem=selem)
    R1 = U / image
    R1 = np.where(image==0, 1, R1)
    index = U > image
    FM = 1 / R1
    FM = np.where(index, R1, FM)
    FM = np.mean(FM)
    return FM


def HISE(image):
    """Histogram entropy

    Reference
    ---------
    [5] Krotkov & Martin (1986).
    """
    selem = disk(WINDOW_SIZE)
    U = rank.entropy(image, selem=selem)
    FM = U.mean()
    return FM


def LAPE(image):
    """Energy of Laplacian

    Reference
    ---------
    [7] Subbarao et al. (1992).
    """
    LAP = np.array([[1/6,   2/3, 1/6],
                    [2/3, -10/3, 2/3],
                    [1/6,   2/3, 1/6]])
    conv = convolve(image.astype(float), LAP, mode='nearest')
    conv = np.clip(conv, a_min=0, a_max=65535)
    FM = np.clip(conv**2, a_min=0, a_max=65535)
    return FM.mean()


def LAPM(image):
    """Modified Laplacian

    Reference
    ---------
    [9] Nayar & Nakagawa (1990).
    """
    M = np.array([[ 0, 0,  0],
                  [-1, 2, -1],
                  [ 0, 0,  0]])
    Lx = convolve(image.astype(float), M, mode='nearest')
    Ly = convolve(image.astype(float), M.T, mode='nearest')
    Lx = np.clip(Lx, a_min=0, a_max=65535)
    Ly = np.clip(Ly, a_min=0, a_max=65535)
    FM = np.abs(Lx) + np.abs(Ly)
    return FM.mean()


def LAPV(image):
    """Variance of Laplacian

    Reference
    ---------
    [6] Pech-Pacheco et al. (2000).
    """
    LAP = np.array([[1/6,   2/3, 1/6],
                    [2/3, -10/3, 2/3],
                    [1/6,   2/3, 1/6]])
    conv = convolve(image.astype(float), LAP, mode='nearest')
    conv = np.clip(conv, a_min=0, a_max=65535)
    return conv.std(ddof=1)**2


def LAPD(image):
    """Diagonal Laplacian

    Reference
    ---------
    [10] Thelen et al. (2008).
    """
    M1 = np.array([[ 0, 0,  0],
                   [-1, 2, -1],
                   [ 0, 0,  0]])
    M2 = np.array([[ 0, 0, -1],
                   [ 0, 2,  0],
                   [-1, 0,  0]]) / np.sqrt(2)
    M3 = np.array([[-1, 0,  0],
                   [ 0, 2,  0],
                   [ 0, 0, -1]]) / np.sqrt(2)
    F1 = convolve(image.astype(float), M1, mode='nearest')
    F1 = np.clip(F1, a_min=0, a_max=65535)
    F2 = convolve(image.astype(float), M2, mode='nearest')
    F2 = np.clip(F2, a_min=0, a_max=65535)
    F3 = convolve(image.astype(float), M3, mode='nearest')
    F3 = np.clip(F3, a_min=0, a_max=65535)
    F4 = convolve(image.astype(float), M1.T, mode='nearest')
    F4 = np.clip(F4, a_min=0, a_max=65535)
    FM = np.abs(F1) + np.abs(F2) + np.abs(F3) + np.abs(F4)
    return FM.mean()


def SFIL(image):
    """Steerable filters

    Reference
    ---------
    [10] Minhas et al. (2009).
    """
    angles = np.pi/180 * np.array([45, 135, 180, 225, 270, 315])
    N = np.floor(WINDOW_SIZE / 2)
    sig = N / 2.5
    y, x = np.mgrid[-N:N+1, -N:N+1]
    G = 1 / (2 * np.pi * sig) * np.exp(-(x ** 2 + y ** 2) / (2 * sig ** 2))
    Gx = -x * G/sig**2
    Gy = -y * G/sig**2
    Gx = Gx / np.sum(np.abs(Gx))
    Gy = Gy / np.sum(np.abs(Gy))
    R = np.zeros(image.shape + (angles.size+2,))
    R[:,:,0] = convolve(image.astype(float), Gx, mode='nearest')
    R[:,:,1] = convolve(image.astype(float), Gy, mode='nearest')
    for i, a in enumerate(angles):
        R[:,:,i+2] = np.cos(a)*R[:,:,0] + np.sin(a)*R[:,:,1]
    FM = np.max(R, axis=2).mean()
    return FM


def SFRQ(image):
    """Spatial frequency

    Reference
    ---------
    [8] Eskicioglu (1995).
    """
    image = image.astype(float)
    Ix = np.zeros_like(image)
    Iy = np.zeros_like(image)
    Ix[:, :-1] = np.clip(image[:, 1:] - image[:, :-1], a_min=0, a_max=None)
    Iy[:-1, :] = np.clip(image[1:, :] - image[:-1, :], a_min=0, a_max=None)
    FM = np.sqrt(Ix ** 2 + Iy ** 2).mean()
    return FM


def TENG(image):
    """Tenengrad

    Reference
    ---------
    [5] Krotkov & Martin (1986).
    """
    Gx = sobel_h(image.astype(float))
    Gy = sobel_v(image.astype(float))
    FM = Gx**2 + Gy**2
    return FM.mean()


def TENV(image):
    """Tenengrad variance

    Reference
    ---------
    [6] Pech-Pacheco et al. (2000).
    """
    Gx = sobel_h(image.astype(float))
    Gy = sobel_v(image.astype(float))
    FM = Gx**2 + Gy**2
    return FM.std(ddof=1)


def VOLA(image):
    """Vollath's correlation

    Reference
    ---------
    [2] Santos et al. (1997).
    """
    image = image.astype(float)
    I1 = image.copy()
    I2 = image.copy()
    I1[:-1, :] = image[1:, :]
    I2[:-2, :] = image[2:, :]
    FM = image * (I1 - I2)
    return FM.mean()
