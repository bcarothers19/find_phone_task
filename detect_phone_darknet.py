"""
Loads the trained YOLO_v2 model and detects a phone in the provided image.
"""
import os
import re
import subprocess

from PIL import Image


def make_darknet():
    """
    Compile the darknet program. We need to run this step first, because the
    compiled file will be different based on the machine it's made on.
    """
    # Get the current working directory
    orig_cwd = os.getcwd()
    # Define the directory where darknet is located. It will be in the same parent
    # directory as the find_phone.py script
    dn_dir = orig_cwd + '/darknet'
    # Check to see if we have already compiled
    if 'darknet' in os.listdir(dn_dir):
        pass
    else:
        print 'Making Darknet...'
        os.chdir(dn_dir)
        # Compile the darknet program
        bash_command = 'make'
        process = subprocess.Popen(bash_command.split(),stdout=subprocess.PIPE)
        output,error = process.communicate()
        os.chdir(orig_cwd)

def detect_phone(image_path):
    """
    Detects the center (x,y) coordinates of a phone. Uses custom trained YOLO_v2
    weights to detect the phone.

    image_path - path of the image we are detecting

    Returns (float,float) - center (x,y) coords of the detected phone, 0<=x,y<=1
    """
    # Compile darknet
    make_darknet()
    # Get the current working directory
    orig_cwd = os.getcwd()
    # Define the directory where darknet is located. It will be in the same parent
    # directory as the find_phone.py script
    dn_dir = orig_cwd + '/darknet'
    os.chdir(dn_dir)
    # Run the test image through our model
    print 'Detecting...'
    # Provide the config file, custom trained weights, and path to image we want
    # to detect
    bash_command = './darknet detect '+dn_dir+'/cfg/phone.cfg '+dn_dir+'/phone.weights '+image_path
    # exit_status = os.system(bash_command)
    # subprocess.call(bash_command,shell=True)
    # print exit_status
    process = subprocess.Popen(bash_command,stdout=subprocess.PIPE,shell=True)
    output,error = process.communicate()
    print output
    os.chdir(orig_cwd)
    # Get the bounding box coordinates
    bb = output.split('Bounding Box: ')[1]
    bb = bb.strip()
    # Parse the bounding box
    x1,y1,x2,y2 = [int(p) for p in bb.split(', ')]

    # Get the original image size
    img_width,img_height = Image.open(image_path).size
    # Calculate the (x,y) of the center of the bounding box
    center_x = ((x1+x2)/2.0)/img_width
    center_y = ((y1+y2)/2.0)/img_height

    return (center_x,center_y)
