import numpy as np
from skimage.util import view_as_windows
from skimage.filters import sobel_h, sobel_v
from scipy.optimize import minimize, Bounds
from scipy.ndimage import convolve
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

# Problem 1: Update H


def shrinkage_operator(ar, epsilon):
    return np.sign(ar)*np.maximum(0, np.abs(ar) - epsilon)


def gradient(image):
    grad_x = sobel_h(image)
    grad_y = sobel_v(image)
    return np.stack((grad_y, grad_x), axis=-1)


def update_H(B, epsilon):
    return shrinkage_operator(gradient(B), epsilon)


# Problem 2: update B and R
def Ps(image, patch_size=(8, 8)):
    patches = view_as_windows(image, patch_size, 1)
    return patches - np.mean(patches, axis=(2, 3), keepdims=True)


def grad_BR_1(x, O):
    v = 2*(x[:x.shape[0]//2] + x[x.shape[0]//2:] - O.flatten())
    return np.concatenate([v, v])


def grad_BR_2(x, beta):
    v = 2*beta*x[x.shape[0]//2:]
    return np.concatenate([np.zeros(x.shape[0]//2), v])


def grad_BR_3(x, H, omega):
    Sh = np.array([[1, 0, -1],
                  [2, 0, -2],
                  [1, 0, -1]])
    Sv = np.array([[1, 2, 1],
                  [0, 0, 0],
                  [-1, -2, -1]])
    grad_B = gradient(x[:x.shape[0]//2].reshape(H.shape[0], H.shape[1]))
    diff = grad_B - H
    return np.concatenate([omega/4*(convolve(diff[:, :, 0], Sv, mode='constant', cval=0.0)
                                    + convolve(diff[:, :, 1], Sh, mode='constant', cval=0.0)).flatten(),
                           np.zeros(x.shape[0]//2)])


def aux_grad_BR_4(diff, a, b, patch_size=(8, 8)):
    p_range = np.arange(
        max(a + 1 - patch_size[0], 0), min(a + 1, diff.shape[0]))
    q_range = np.arange(
        max(b + 1 - patch_size[1], 0), min(b + 1, diff.shape[1]))
    p_grid, q_grid = np.meshgrid(p_range, q_range, indexing='ij')
    idx2 = a - p_grid
    idx3 = b - q_grid
    return np.sum(diff[p_grid, q_grid, idx2, idx3])


def grad_BR_4(x, gB, gR, omega, image_shape, patch_size=(8, 8)):
    B = x[:x.shape[0]//2].reshape(image_shape)
    R = x[x.shape[0]//2:].reshape(image_shape)
    B_patches, R_patches = Ps(B, patch_size), Ps(R, patch_size)
    diff_B = B_patches - gB
    diff_R = R_patches - gR

    grad_B = -2*omega/(patch_size[0]*patch_size[1])\
        * np.array([np.sum(diff_B[a+1-patch_size[0]:a+1, b+1-patch_size[1]:b+1].flatten())
                    for a in range(image_shape[0]) for b in range(image_shape[1])]) \
        + 2*omega * np.array([aux_grad_BR_4(diff_B, a, b, patch_size)
                              for a in range(image_shape[0]) for b in range(image_shape[1])])

    grad_R = -2*omega/(patch_size[0]*patch_size[1])\
        * np.array([np.sum(diff_R[a+1-patch_size[0]:a+1, b+1-patch_size[1]:b+1].flatten())
                    for a in range(image_shape[0]) for b in range(image_shape[1])])\
        + 2*omega * np.array([aux_grad_BR_4(diff_R, a, b, patch_size)
                              for a in range(image_shape[0]) for b in range(image_shape[1])])

    return np.concatenate([grad_B, grad_R])


def grad_BR(x, image, gB, gR, H, beta, omega, patch_size=(8, 8)):
    return grad_BR_1(x, image) \
        + grad_BR_2(x, beta) \
        + grad_BR_3(x, H, omega) \
        + grad_BR_4(x, gB, gR, omega, image.shape, patch_size)


def obj_BR(x, image, gB, gR, H, beta, omega):
    B = x[:x.shape[0]//2]
    R = x[x.shape[0]//2:]
    B_patches = Ps(B.reshape(image.shape))
    R_patches = Ps(R.reshape(image.shape))
    grad_B = gradient(B.reshape(image.shape))
    return np.linalg.norm(image.flatten() - B - R)**2\
        + beta*np.linalg.norm(R)**2\
        + omega*np.linalg.norm(grad_B.flatten() - H.flatten())**2\
        + omega*np.linalg.norm(B_patches.flatten() - gB.flatten())**2\
        + omega*np.linalg.norm(R_patches.flatten()-gR.flatten())**2


def update_BR(image, B, R, gB, gR, H, beta, omega):
    minim = minimize(obj_BR, np.concatenate([B.flatten(), R.flatten()]), (image, gB, gR, H, beta, omega),
                     'L-BFGS-B', grad_BR, bounds=Bounds(lb=0)).x
    B = minim[:minim.shape[0]//2].reshape(image.shape)
    R = minim[minim.shape[0]//2:].reshape(image.shape)
    return B, R


# Problem 3: update g
def most_likely_classes(patches, gmm):
    return gmm.predict(patches.reshape(-1, patches.shape[2]*patches.shape[3]))\
        .reshape(patches.shape[0], patches.shape[1])


def update_g(image, gmm, omega, gamma):
    sigma_2 = gamma/(2*omega)
    patches = Ps(image)
    labels = most_likely_classes(patches, gmm)

    covs, means = gmm.covariances_[labels], gmm.means_[labels]
    A = np.linalg.inv(
        covs + sigma_2*np.eye(patches.shape[-1]*patches.shape[-2]))
    B = np.matmul(covs,
                  patches.reshape(patches.shape[0], patches.shape[1], patches.shape[2]*patches.shape[3], 1))\
        + sigma_2*means[..., None]
    g = np.matmul(A, B)
    return g.reshape(patches.shape)


def opti(image,
         r_gmm, b_gmm,
         alpha=1e-1, beta=1e-2, gamma=1e-1,
         omega0=1e-3,
         epsilon=1e-10, max_iter=20,
         B0=None, R0=None, gB0=None, gR0=None):
    B = image.copy() if B0 is None else B0
    R = np.zeros(B.shape) if R0 is None else R0
    gB = Ps(B) if gB0 is None else gB0
    gR = Ps(R) if gR0 is None else gR0
    H = gradient(B)
    omega = omega0

    for _ in tqdm(range(max_iter)):
        newH = update_H(B, alpha/(2*omega))
        newB, newR = update_BR(image, B, R, gB, gR, H, beta, omega)
        newgR = update_g(newR, r_gmm, omega, gamma)
        newgB = update_g(newB, b_gmm, omega, gamma)
        omega *= 2

        if np.linalg.norm(newB - B) + np.linalg.norm(newR - R) < epsilon \
                and np.linalg.norm(newH - H) < epsilon \
                and np.linalg.norm(newgB - gB) < epsilon \
                and np.linalg.norm(newgR - gR) < epsilon:
            B, R = newB, newR
            H = newH
            gB, gR = newgB, newgR
            break
        B, R = newB, newR
        H = newH
        gB, gR = newgB, newgR
    return B, R, gB, gR
