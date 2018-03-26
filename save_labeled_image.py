"""
Gets the output image (with bounding box and class label) produced by Darknet
when running find_phone.py.

Adds a dot at the center of the detected phone.

Save the image to the tested_images/ directory.
"""
import os

from PIL import Image, ImageDraw


def load_image():
    """
    Loads the output image with detected bounding box.

    Returns an image in PIL format
    """
    # Get the current working directory
    cwd = os.getcwd()
    # Define the directory where darknet is located. It will be in the same parent
    # directory as the find_phone.py script
    dn_dir = cwd + '/darknet/'
    # The output image with bounding box will always be titled predictions.png
    output_img_path = dn_dir + 'predictions.png'
    # Load the image
    output_img = Image.open(output_img_path)

    return output_img

def add_center_xy(img,x,y):
    """
    Add a small circle at the detected center (x,y) coordinate.

    img - PIL image
    x,y -  coordinate position scaled 0 <= x,y <= 1

    Returns the annotated image with a dot added
    """
    # Get the width and height of the original image
    width,height = img.size
    # Calculate the actual (x,y) pixel coordinates
    x_px = float(x) * width
    y_px = float(y) * height
    # Draw a green circle at the center of the detected phone
    draw = ImageDraw.Draw(img)
    draw.ellipse((x_px-2, y_px-2, x_px+2, y_px+2), fill=(0,255,0,255))

    return img


def save_image(x,y,orig_fn):
    """
    Adds a dot to the center of the output annotated image, and save it in a
    folder for processed images.

    x,y - coordinate position scaled 0 <= x,y <= 1
    orig_fn - filename of the original input image
    """
    # Load the image
    detected_image = load_image()
    # Add a dot at the center (x,y) coordinate
    detected_image = add_center_xy(detected_image,x,y)
    # Create the filepath where we will save the modified image
    new_fn = os.path.split(orig_fn)[1]
    new_fn = os.getcwd() + '/tested_images/' + new_fn
    # Save the image
    detected_image.save(new_fn)
