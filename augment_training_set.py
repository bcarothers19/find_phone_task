"""
Augments the original image set by transforming the image and its bounding box.

Saves the transformed image and bounding box in the augmented_images/ directory.
"""
import os
import re

import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
from PIL import Image
from keras.preprocessing import image

ia.seed(87)

# Get the current working directory
cwd = os.getcwd()
# Define the paths to the folders we will use
bb_label_folder = cwd + '/bbox_labels/' # Labels created with BBox Label Tool
img_folder = cwd + '/find_phone/' # Original images
augmented_img_folder = cwd + '/augmented_images/' # Will contain output images


def load_bb(img_num, img):
    """
    Loads the bounding box for a given image.

    img_num - the image number (filename identifier)
    img - numpy array representing an image

    Returns an imgaug bounding box object
    """
    # Read in the bounding box text file
    with open(bb_label_folder+str(img_num)+'.txt') as f:
        bb_text = f.read()
    # Parse the coordinates of the bounding box
    coords = bb_text.split('\n')[1].split(' ')
    coords = [int(x) for x in coords]
    # Create an imgaug bounding box
    bb = ia.BoundingBoxesOnImage([ia.BoundingBox(x1=coords[0], y1=coords[1], x2=coords[2], y2=coords[3])], shape=img.shape)
    return bb

def load_img(img_num):
    """
    Loads an image and its corresponding bounding box

    img_num - the image number (filename identifier)

    Returns the image and its bounding box
    """
    # Define the image filename
    img_name = ''.join([str(img_num),'.jpg'])
    # Image size is relatively smalL and we have a small training set, so load
    # the image in as it's original size
    img = image.load_img(img_folder + img_name, target_size = (326,490))
    # Convert the image to a numpy array
    img = image.img_to_array(img)
    # Load the bounding box for this image
    bb = load_bb(img_num,img)

    return (img,bb)

def transform_image(image,bb,sequence):
    """
    Transforms a image and its bounding box with the given tranformation sequence.

    image - numpy array (representing an image)
    bb - imgaug bounding box
    sequence - a list of imgaug tranformations

    Returns the new transformed image and bounding box
    """
    # Define the sequence
    seq = iaa.Sequential(sequence)
    # Set the sequence to be deterministic so the same tranformation is applied
    # to the image and bounding box
    seq_det = seq.to_deterministic()
    # Transform the image and bounding box
    images_aug = seq_det.augment_images([image])
    bbs_aug = seq_det.augment_bounding_boxes([bb])

    return (images_aug[0],bbs_aug[0].bounding_boxes[0])


# Define the sequences we will use, and an identifier to append to the filename
sequences = [([iaa.Fliplr(1)],'_lr'),([iaa.Flipud(1)],'_ud'),
             ([iaa.Fliplr(1),iaa.Flipud(1)],'_udlr')]

# Iterate through all of the images in the original folder
for fn in os.listdir(img_folder):
    if '.jpg' in fn:
        # Perform each of the defined sequence transformations
        for sequence,identifier in sequences:
            # Load the image and bounding box
            image_number = fn.split('.')[0]
            img,bb = load_img(image_number)
            # Transform the image and bounding box with this sequence
            new_img,new_bb = transform_image(img,bb,sequence)
            # Define the new filename
            new_fn = image_number + identifier
            # Save the transformed image
            Image.fromarray(new_img.astype('uint8')).save(augmented_img_folder + new_fn + '.jpg')
            # Save the new bounding box coordinates
            bb_string = ' '.join([str(new_bb.x1),str(new_bb.y1),str(new_bb.x2),str(new_bb.y2)])
            with open(augmented_img_folder + new_fn + '.txt','w') as f:
                f.write('0\n' + bb_string)
