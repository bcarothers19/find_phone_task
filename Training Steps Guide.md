# Phone Detection Training Process
To complete this object detection challenge, I implemented the YOLO_v2 architecture. This required additional steps such as re-labeling the images, augmenting the data set, and configuring and customizing Darknet. While all of this could be converted into a Python script which relies heavily on shell scripting, that would not be representative of a typical pipeline. Training the model also took over 14 hours using a GPU, which is not reasonable to expect you to re-train. Instead, I wrote this guide detailing the steps of working with the provided images and training the model. All packages and scripts used in this process are included in the submitted project file. The trained model weights are included and are loaded by `find_phone.py` to detect the center (x,y) coordinates of a new image.

### Label Images
The YOLO_v2 architecture requires a bounding box as the label during training. Since the provided data only contains the center (x,y) coordinate of the phone, we need to re-annotate the data set. I used the [BBox Label Tool](https://github.com/puzzledqs/BBox-Label-Tool) to annotate the images.

![alt text](https://s3.amazonaws.com/find-phone-bcarothers/bbox_label_screenshot.png "BBox Label Tool")

### Augment Images
The original data set size of 129 images is small for an object detection task. To increase the number and diversity of images in our training set, we can perform image augmentation on the original images. This involves techniques such as flipping, rotating, blurring, changing the image levels, etc. The Python package [imgaug](https://github.com/aleju/imgaug) makes this task a lot easier. It allows us to load an image as well as its annotated bounding boxes and performs the same chain of transformations on each - so it automatically calculates the new bounding box for the transformed image.
![alt text](https://s3.amazonaws.com/find-phone-bcarothers/flipped_sample.png "Sample image flipped horizontally and vertically")
It is recommended to have at least 300 images for each object you are trying to detect, but having over 1000 per class is preferred. Images that don't contain the object we're trying to detect can also improve the training of the model.

The script `augment_training_set.py` was used to create the augmented images and their corresponding bounding box txt file.

### Convert Annotation Format
The YOLO_v2 architecture expects training labels in the format

`class_num center_x center_y width height`

where the last 4 values are scaled to be between 0 and 1. We need to convert the output of the BBox Label Tool to fit this format. The original annotation format is
```
class_num
x1 y1 x2 y2
```
We use the script `convert_annotations.py`, which was modified from [Guanghan's Darknet fork.](https://github.com/Guanghan/darknet/blob/master/scripts/convert.py)

### Setup Darknet
###### Arrange Data Set
We need to arrange our images and labels in the proper way for Darknet to read them during training. We create two folders:`data/phone/images/` and `data/phone/labels/`, where each file such as `87.jpg` in `images` has a corresponding `87.txt` in `labels`.

###### Split into training and testing sets
We divide the images into the training and testing sets by creating two files (`train.txt` and `test.txt`), which are lists of the filepath for each image in their respective set. Darknet will automatically find the corresponding label file to load during training. These txt files are later provided to a Darknet configuration file. Sample lines from `train.txt`:
```
data/phone/images/0.jpg
data/phone/images/117.jpg
data/phone/images/54.jpg
```
We use the script `split_images.py` to create these files.

###### Configure Darknet Files
We have three files to set up to configure Darknet:
1. `data/phone.names`
This file contains the class that corresponds to each class_num in our labels. There is one class per line starting at index 0, but since we are only detecting phones our `phone.names` is simple:

	```phone```

2. `cfg/phone.data`
This file contains information which tells Darknet where to look for certain files. We define the number of classes, paths to `train.txt`, `test.txt`, and `phone.names`, as well as a folder where model weights will be saved at predefined numbers of batches completed. For this use case, our `phone.data` file will look like:
	```
	classes = 1
	train = data/phone/train.txt
	valid = data/phone/test.txt
	names = data/phone.names
	backup = backup/
   ```

3. `cfg/phone.cfg`
This file defines the architecture, input structure, and output structure of our model. We start with the base `yolo.2.0.cfg` file and make some changes. I changed the input image width and height to 608 instead of 416 - increased image size generally leads to improved model performance. We also need to define the number of classes, which is 1 in our case. The number of filters in the final convolutional layer is dependent on our number of classes (`filters = (classes + coords + 1) * num`), which in our case is `(1 + 4 + 1) * 5 = 30`. In summary, the changes made to this file are:
	```
    [net]
	width=608
	height=608

	[convolutional]
    filters-30

    [region]
    classes=1
    ```

###### Download Weights File
The final item we need is a weights file to use as our initial model weights. We can use the `darknet19_448.conv.23` file which can be [downloaded here](https://pjreddie.com/media/files/darknet19_448.conv.23). This file should be placed in the main Darknet directory.


### Training YOLO_v2 with Darknet
Now that we have our configuration files created and everything in their proper directories, we can begin training our model. I used an AWS EC2 p2.xlarge (GPU compute optimized) instance to train the model. After properly configuring CUDA and CUDNN, we are able to run the computations on the K80 GPU. To being training, we call this command from the Darknet directory:

`./darknet detector train cfg/phone.data cfg/phone.cfg darknet19_448.conv.23`

After training for ~14 hours, we have completed over 6000 batches and we can see that our model has an average loss of 0.198352. The typical Intersection over Union is 80-90% and the average recall is usually 100%. It takes about 9 seconds to complete one batch of 64 images.

![alt text](https://s3.amazonaws.com/find-phone-bcarothers/Training+Output.png "Training output")

After 6000 batches I tested the trained model with the held out set of images. On two separate testing sets, we obtained average normalized distance from the true center of `0.00524` and `0.00638`. The model also performed well on out-of-sample images that I took of my phone in various locations to test the robustness of the model. While there is still room for improvement (detailed in `Next Steps.md`) we will stop training for now, since the model is able to detect phones and find its center with a high degree of accuracy.

### Sample images
Detected phones with bounding box and center (x,y) coordinates displayed on the image. These images can be found in `tested_images/` after running `find_phone.py`.
###### Testing images
![](https://s3.amazonaws.com/find-phone-bcarothers/9.jpg)
![](https://s3.amazonaws.com/find-phone-bcarothers/71.jpg)
![](https://s3.amazonaws.com/find-phone-bcarothers/81.jpg)

###### Out-of-sample images
![](https://s3.amazonaws.com/find-phone-bcarothers/IMG_0017.JPG)
![](https://s3.amazonaws.com/find-phone-bcarothers/IMG_0008.JPG)
