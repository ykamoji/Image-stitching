import os
import time
from utils import imread
from detectBlobs import detectBlobs
from drawBlobs import drawBlobs


# This code is part of:
#
#   CMPSCI 670: Computer Vision, Spring 2024
#   University of Massachusetts, Amherst
#   Instructor: Grant Van Horn
#

# Evaluation code for blob detection
# Your goal is to implement scale space blob detection using LoG

class Params:
    def __init__(self, levels=10, initial_sigma=2, k=2**0.35, threshold=0.001):
        self.levels = levels
        self.initial_sigma = initial_sigma
        self.k = k
        self.threshold = threshold

    def set_filter_method(self, filter):
        self.filter = filter


## LOG
LoG_paramsMap = {
    'butterfly.jpg': Params(),
    'einstein.jpg': Params(),
    'faces.jpg': Params(),
    'fishes.jpg': Params(),
    'football.jpg': Params(threshold=0.0005),
    'sunflowers.jpg': Params()
}

## DOG
DoG_paramsMap = {
    'butterfly.jpg': Params(threshold=0.00015),
    'einstein.jpg': Params(threshold=0.0001),
    'faces.jpg': Params(threshold=0.0005),
    'fishes.jpg': Params(threshold=0.0001),
    'football.jpg': Params(threshold=0.0001),
    'sunflowers.jpg': Params(threshold=0.00015)
}

numBlobsToDraw = 1000
images = ['butterfly.jpg', 'einstein.jpg', 'faces.jpg', 'fishes.jpg', 'football.jpg', 'sunflowers.jpg']

results = {}
datadir = os.path.join('..', 'data', 'blobs')
for imageName in images:
    imName = imageName.split('.')[0]
    im = imread(os.path.join(datadir, imageName))
    print(f"Detecting blobs for {imName}:")
    results[imName] = {}
    for filter, paramsMap in zip(["LOG","DOG"],[LoG_paramsMap, DoG_paramsMap]):
        params = paramsMap[imageName]
        params.set_filter_method(filter)
        start = time.time()
        blobs = detectBlobs(im, params)
        results[imName][filter] = (params, len(blobs), time.time() - start)
        # print(f"Time taken = {(time.time() - start):0.4f}")
        drawBlobs(im, blobs, numBlobsToDraw, imName, params, save_path="../output/blob_detector/")


print("\nCompleted !\n")
print("Image"+"\t"*2+"Filter"+"\t"*2+"Parameters"+"\t"*6+"Blobs"+"\t"*2+"Time")
print("-"*120)


def print_params(params):
    return f"level={params.levels}, sigma={params.initial_sigma}, k={params.k:.3f}, threshold={params.threshold}"


for image_name, data in results.items():
    for filter, (params, blob_count, duration) in data.items():
        print(f"{image_name.ljust(15, ' ')}"+ "\t" + f"{filter}" + "\t" * 2 +
              f"{print_params(params)}" + "\t" * 2 + f"{blob_count}" + "\t" * 2 + f"{duration:.3f}")
print("-" * 120)

