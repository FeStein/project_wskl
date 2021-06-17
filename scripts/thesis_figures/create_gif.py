import os
import sys
IMAGE_FOLDER="/home/felix/Documents/output"

filenames = sorted(os.listdir(IMAGE_FOLDER), key = lambda x: int(x.split('.')[0][4:]))

import imageio
images = []

for filename in filenames[20:]:
    path = os.path.join(IMAGE_FOLDER, filename)
    images.append(imageio.imread(path))
imageio.mimsave('sample.gif', images)
