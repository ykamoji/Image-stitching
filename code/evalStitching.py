import numpy as np
import matplotlib.pyplot as plt
import os
from utils import imread
from utils import showMatches
from detectBlobs import detectBlobs
from computeSift import compute_sift
from computeMatches import computeMatches
from ransac import ransac
from mergeImages import mergeImages


# This code is part of:
#
#   CMPSCI 670: Computer Vision, Spring 2024
#   University of Massachusetts, Amherst
#   Instructor: Grant Van Horn
#

class Params:
    def __init__(self, levels=10, initial_sigma=2, k=2**0.35, threshold=0.001):
        self.levels = levels
        self.initial_sigma = initial_sigma
        self.k = k
        self.threshold = threshold

    def set_filter_method(self, filter):
        self.filter = filter


# Image directory
dataDir = os.path.join('..', 'data', 'stitching')

# Read input images
testExamples = ['stop', 'car', 'building', 'book', 'eg', 'house1', 'house2', 'kitchen', 'park', 'pier', 'roof', 'table']

output_dir = "../output/imageStitching/"

paramsMap = {
    'stop': Params(threshold=0.002), # Validated
    'car': Params(threshold=0.002), # Validated
    'building': Params(threshold=0.0005), # Validated
    'book': Params(threshold=0.0005), # Validated
    'eg': Params(),  # Validated
    'house1': Params(),  # Validated
    'house2': Params(), # Validated
    'kitchen': Params(threshold=0.00001), # Best possible with maxInlierError = 40,
                                            # randomSeedSize & goodFitThresh = 5
    'park': Params(threshold=0.0005), # Validated
    'pier': Params(), # Validated
    'roof': Params(threshold=0.0001), # Validated
    'table': Params(threshold=0.00001), # Validated
}

for example in testExamples:
    print(f"Stitching {example}...")
    output_path = output_dir + example + '/'

    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    imageName1 = '{}_1.jpg'.format(example)
    imageName2 = '{}_2.jpg'.format(example)

    im1 = imread(os.path.join(dataDir, imageName1))
    im2 = imread(os.path.join(dataDir, imageName2))

    params = paramsMap[example]
    params.set_filter_method('DOG')

    # Detect keypoints
    print(f"Finding blobs...")

    blobs1 = detectBlobs(im1, params)
    blobs2 = detectBlobs(im2, params)

    # Compute SIFT features
    sift1 = compute_sift(im1, blobs1[:, :4])
    sift2 = compute_sift(im2, blobs2[:, :4])

    print(f"Finding matches and running ransac...")
    # Find the matching between features
    matches = computeMatches(sift1, sift2)
    showMatches(im1, im2, blobs1, blobs2, matches, title="Matches", output_path=output_path + example + '_matches')
    # Ransac to find correct matches and compute transformation
    inliers, transf = ransac(matches, blobs1, blobs2)

    goodMatches = np.full_like(matches, -1)
    goodMatches[inliers] = matches[inliers]

    showMatches(im1, im2, blobs1, blobs2, goodMatches, title="Good matches", output_path=output_path + example +
                                                                                         '_g_matches')
    transf = np.linalg.inv(np.vstack((transf, [0, 0, 1])))[:2,:]
    trans_print = np.round(transf, 4)
    print(f"Estimated affine transformation:\n{trans_print[:1,:].tolist()[0]}\n{trans_print[-1:,:].tolist()[0]}")

    # Merge two images and display the output
    stitchIm = mergeImages(im1, im2, transf)
    plt.figure()
    plt.imshow(stitchIm)
    plt.title('Stitched image: {}'.format(example))
    plt.axis('off')
    plt.savefig(output_path + example + '_stitched', bbox_inches='tight', edgecolor='auto')
    plt.show()

    print(f"Completed!\n")
