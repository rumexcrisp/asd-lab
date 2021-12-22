#!/usr/bin/env python3

import numpy as np


def LS_lane_fit(pL, pR):
    """
    LS estimate for lane coeffients z=(W, Y_offset, Delta_Phi, c0)^T.

    Args:
        pL: [NL, 2]-array of left marking positions (in DIN70000)
        pR: [NR, 2]-array of right marking positions (in DIN70000)

    Returns:
        Z: lane coeffients (W, Y_offset, Delta_Phi, c0)
    """

    H = np.zeros((pL.shape[0] + pR.shape[0], 4))  # design matrix
    Y = np.zeros((pL.shape[0] + pR.shape[0], 1))  # noisy observations

    # fill H and Y for left line points
    for i in range(pL.shape[0]):
        u, v = pL[i, 0], pL[i, 1]
        u2 = u * u
        H[i, :] = [0.5, -1, -u, 1.0 / 2.0 * u2]
        Y[i] = v

    # fill H and Y for right line points
    for i in range(pR.shape[0]):
        u, v = pR[i, 0], pR[i, 1]
        u2 = u * u
        u3 = u2 * u
        H[pL.shape[0] + i, :] = [-0.5, -1, -u, 1.0 / 2.0 * u2]
        Y[pL.shape[0] + i] = v

    # compute optimal state vector Z
    Z = np.dot(np.linalg.pinv(H), Y)

    return Z


def LS_lane_compute(Z, maxDist=60, step=0.5):
    """
    Compute lane points from given parameter vector.

    Args;
        Z: lane coeffients (W, Y_offset, Delta_Phi, c0)
        maxDist[=60]: distance up to which lane shall be computed
        step[=0.5]: step size in x-direction (in m)

    Returns:
        (x_pred, yl_pred, yr_pred): x- and y-positions of left and
            right lane points
    """
    x_pred = np.arange(0, maxDist, step)
    yl_pred = np.zeros_like(x_pred)
    yr_pred = np.zeros_like(x_pred)

    for i in range(x_pred.shape[0]):
        u = x_pred[i]
        u2 = u * u
        yl_pred[i] = np.dot(np.array([0.5, -1, -u, 1.0 / 2.0 * u2]), Z)
        yr_pred[i] = np.dot(np.array([-0.5, -1, -u, 1.0 / 2.0 * u2]), Z)

    return (x_pred, yl_pred, yr_pred)


def LS_lane_residuals(lane_left, lane_right, Z):
    residual = np.zeros((lane_left.shape[0] + lane_right.shape[0], 1))
    for i in range(lane_left.shape[0]):
        u, v = lane_left[i, 0], lane_left[i, 1]
        u2 = u * u
        residual[i] = np.dot(np.array([0.5, -1, -u, 1.0 / 2.0 * u2]), Z) - v

    for i in range(lane_right.shape[0]):
        u, v = lane_right[i, 0], lane_right[i, 1]
        u2 = u * u
        u3 = u2 * u
        residual[lane_left.shape[0] + i] = (
            np.dot(np.array([-0.5, -1, -u, 1.0 / 2.0 * u2]), Z) - v
        )

    return residual


def LS_lane_inliers(residual, thresh):
    inlier = residual[np.abs(residual) < thresh]
    return inlier.shape[0]


def Cauchy(r, sigma=1):
    """
    Cauchy loss function.

    Args:
        r: resiudals
        sigma: expected standard deviation of inliers

    Returns:
        w: vector of weight coefficients
    """
    c = 2.3849 * sigma
    wi = np.zeros(len(r))
    for i in range(len(r)):
        wi[i] = 1 / (1 + np.power((r[i]) / c, 2))
    return wi


def MEstimator_lane_fit(pL, pR, Z_initial, sigma=1, maxIteration=10):
    """
    M-Estimator for lane coeffients z=(W, Y_offset, Delta_Phi, c0)^T.

    Args:
        pL: [NL, 2]-array of left marking positions (in DIN70000)
        pR: [NR, 2]-array of right marking positions (in DIN70000)
        Z_initial: the initial guess of the parameter vector
        sigma: the expecvted standard deviation of the inliers
        maxIteration: max number of iterations

    Returns:
        Z: lane coeffients (W, Y_offset, Delta_Phi, c0)
    """

    H = np.zeros((pL.shape[0] + pR.shape[0], 4))  # design matrix
    Y = np.zeros((pL.shape[0] + pR.shape[0], 1))  # noisy observations

    # fill H and Y for left line points
    for i in range(pL.shape[0]):
        u, v = pL[i, 0], pL[i, 1]
        u2 = u * u
        H[i, :] = [0.5, -1, -u, 1.0 / 2.0 * u2]
        Y[i] = v

    # fill H and Y for right line points
    for i in range(pR.shape[0]):
        u, v = pR[i, 0], pR[i, 1]
        u2 = u * u
        u3 = u2 * u
        H[pL.shape[0] + i, :] = [-0.5, -1, -u, 1.0 / 2.0 * u2]
        Y[pL.shape[0] + i] = v

    Z = Z_initial
    for i in range(maxIteration):

        # store old data
        Z0 = Z

        # compute residuals
        res = LS_lane_residuals(pL, pR, Z)

        # recompute weights
        W = np.diag(Cauchy(res, sigma))

        # recompute new estimate
        HTWH = np.dot(np.dot(H.T, W), H)
        inv_HTWH = np.linalg.inv(HTWH)
        Z = np.dot(np.dot(inv_HTWH, H.T), np.dot(W, Y))

        # print('iter %d: ' % i, Z.T)

    return Z
