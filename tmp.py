import matplotlib.pyplot as plt
import cv2, os
from skimage.feature import hog
from skimage import exposure
import numpy as np

path = os.path.expanduser("~/Pictures/IMG_2316.JPG")
image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)

image[np.where(image>128)] = 255
image[np.where(image<=128)] = 0

image = np.tile(np.expand_dims(image, axis=-1), (1, 1, 3))



fd, hog_image = hog(image, orientations=16, pixels_per_cell=(4, 4),
                    cells_per_block=(1, 1), visualize=True, multichannel=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
plt.savefig(os.path.expanduser("~/Pictures/result.jpg"))
plt.show()