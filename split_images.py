"""
Splits the images located in data/phone/images/ into training and testing sets.

Creates train.txt and test.txt located in data/phone/.
"""
import os
import glob

# Get the current directory
cwd = os.getcwd()
# Directory where the data will reside, relative to the current directory
path_data = '/darknet/data/phone/images/'
# Create absolute path to image folder
path_data = ''.join([cwd,path_data])

# Percentage of images to be used for the test set
percentage_test = 10;

# Create and/or truncate train.txt and test.txt
file_train = open('train.txt', 'w')
file_test = open('test.txt', 'w')

# Populate train.txt and test.txt
counter = 1
index_test = round(100 / percentage_test)
# Iterate through all files in the image folder
for pathAndFilename in glob.iglob(os.path.join(path_data, "*.jpg")):
    # Get the image name
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))
    # Determine if image will be in the training or testing set
    if counter == index_test:
        counter = 1
        file_test.write(path_data + title + '.jpg' + "\n")
    else:
        file_train.write(path_data + title + '.jpg' + "\n")
        counter = counter + 1
