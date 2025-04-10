import numpy as np


# This code is part of:
#
#   CMPSCI 670: Computer Vision, Spring 2024
#   University of Massachusetts, Amherst
#   Instructor: Grant Van Horn
#

def computeAffineMatrix(points1, points2):
    numPoints = len(points1)
    P1 = np.concatenate([points1.T, np.ones([1, numPoints])], axis=0)
    P2 = np.concatenate([points2.T, np.ones([1, numPoints])], axis=0)

    P1_T = np.transpose(P1)
    P2_T = np.transpose(P2)

    H_T = np.linalg.lstsq(P1_T, P2_T, rcond=None)[0]
    H = np.transpose(H_T)
    H[2, :] = [0, 0, 1]

    return H


def compute_error(H, pt1, pt2, match):
    pt1_match = pt1[match[:, 0], :]
    pt2_match = pt2[match[:, 1], :]
    N = match.shape[0]

    pt1_homo = np.concatenate([pt1_match.T, np.ones([1, N])], axis=0)
    pt2_homo = np.concatenate([pt2_match.T, np.ones([1, N])], axis=0)

    pt1_trans = np.dot(H, pt1_homo)

    error = pt1_trans[:2, :] - pt2_homo[:2, :]
    L2_norm = np.sqrt(np.sum(error * error, axis=0))
    dists = L2_norm.reshape((-1, 1))
    dists = dists.flatten()

    return dists


def part(pointMatches, splitSize):
    randIdx = np.random.permutation(len(pointMatches))
    set1 = pointMatches[randIdx[0:splitSize], :]
    set2 = pointMatches[randIdx[splitSize:], :]
    return set1, set2


def ransac(matches, blobs1, blobs2):
    points1 = blobs1[:, :2]
    points2 = blobs2[:, :2]
    points_num = matches.shape[0]
    pointMatches = []
    for ind in range(points_num):
        if matches[ind] != -1:
            pointMatches.append([ind, matches[ind]])

    pointMatches = np.array(pointMatches)
    match_num = len(pointMatches)
    randomSeedSize = np.max([np.ceil(0.5 * match_num), 3])
    goodFitThresh = np.floor(0.85 * match_num)

    T = np.eye(3)
    error = np.inf
    random_points = int(randomSeedSize)
    maxInlierError = 20
    inliers = []
    for i in range(2000):
        set1, set2 = part(pointMatches, random_points)
        temp_T = computeAffineMatrix(points1[set1[:, 0], :], points2[set1[:, 1], :])
        temp_error = compute_error(temp_T, points1, points2, set2)
        valid_pts = temp_error <= maxInlierError
        if np.sum(valid_pts) + random_points >= goodFitThresh:
            valid_set = np.concatenate([set1, set2[valid_pts, :]], axis=0)
            temp_T = computeAffineMatrix(points1[valid_set[:, 0], :], points2[valid_set[:, 1], :])
            final_error = np.sum(compute_error(temp_T, points1, points2, valid_set))

            if final_error < error:
                # print('Found better transformation !')
                T = temp_T
                inliers = set1
                error = final_error

    best_inliers_ptx = [-1] * points_num
    for ind1, ind2 in inliers:
        best_inliers_ptx[ind1] = ind2

    inliners_idx = np.intersect1d(matches, best_inliers_ptx, return_indices=True)[1]
    return inliners_idx, T[:2, :]
