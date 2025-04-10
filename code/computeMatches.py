import numpy as np
from scipy.spatial.distance import cdist

# This code is part of:
#
#   CMPSCI 670: Computer Vision, Spring 2024
#   University of Massachusetts, Amherst
#   Instructor: Grant Van Horn
#

def computeMatches(f1, f2):
    """ Match two sets of SIFT features f1 and f2 """
    matches = []
    for i in range(len(f1)):
        min_index, min_distance = -1, np.inf
        sec_index, sec_distance = -1, np.inf

        for j in range(len(f2)):
            distance = np.linalg.norm(f1[i] - f2[j])

            if distance < min_distance:
                sec_index, sec_distance = min_index, min_distance
                min_index, min_distance = j, distance

            elif distance < sec_distance and sec_index != min_index:
                sec_index, sec_distance = j, distance

        matches.append([min_index, min_distance, sec_distance])

    good_matches = []
    for i in range(len(matches)):
        good_matches.append( matches[i][0] if matches[i][1] <= matches[i][2] * 0.8 else -1)

    good_matches = np.array(good_matches)
    return good_matches




