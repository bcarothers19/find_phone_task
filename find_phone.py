"""
Detects a phone in a provided image.

Accepts a command line argument of a path to an image.

Prints the center (x,y) coordinate of the detected phone in format 'x y'

Saves an annotated image in tested_images/.
"""
import os
import sys

import detect_phone_darknet
import save_labeled_image

# Parse the input image name
try:
    image_path = str(sys.argv[1])
except:
    raise IOError("Must provide a path to an image")

try:
    # Detect the phone and print its center (x,y) coordinates
    x,y = detect_phone_darknet.detect_phone(image_path)

    # Print the (x,y) coordinates (where 0 <= x,y <= 1)
    print ' '.join([str(x),str(y)])

    # Modify the annoted image and save it in the tested_images/ directory
    save_labeled_image.save_image(x,y,image_path)
except IndexError:
    # No phone was detected above the defined threshold, so there won't be a
    # bounding box
    print 'Phone not detected above confidence threshold'
