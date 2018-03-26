
Although my model obtained a high degree of accuracy, I believe there are still several areas for improvement. The main areas I would focus on is improving the robustness of the model (different types of phones, more diverse environments, multiple phones in one image, different lighting conditions, etc.), and improving the speed of processing.

# Next Steps:

### More images in the data set

I would want to have a larger set of initial images, particularly more diverse images. Additional images that included different models of phones, in different backgrounds, and different angles would help to improve the robustness of the model and increase overall accuracy. On the additional images that I took of my phone to test, the model still detected the phone accurately but classified the object as a phone with less confidence (typically ~90% on original images, sometimes ~50% on new images). A more diverse training set would increase accuracy (detection and classification) on these images.


### More augmented images

I would train the model with more augmented images, even with a larger set of original images. Specifically, I would like to include images modified with blur, sharpness, noise, and inverted colors. The set used to train the model actually only contains ~360 images (although Darknet performs some image augmentation as well while training), with the original image and horizontally and vertically flipped copies of each image. I intended to test the model architecture and get a baseline score with this small set of data, but after letting it train overnight it performed beyond my expectations and I decided to pause training (mainly due to the narrow scope of this task). If this model were being implemented in production, I would continue training with more varied data sets.

### Longer training

I would also like to continue training the model for more batches. As I noted above, I paused training at 6000 batches. Normally, I would train for a minimum of 10,000 iterations and evaluate the training history at that point.

### Experiment with model architecture

I would like to experiment with different model architectures (Fast R-CNN, Tiny YOLO, smaller input image sizes, etc.). I want to explore the trade-off between accuracy and speed with different architectures before deciding on the final model and fine-tuning.

### Analyze and fine-tune

I would spend more time analyzing the training metrics and fine-tuning the model. In addition to training longer to minimize training scores, I’d also evaluate the testing scores at different batch counts (i.e. 5000, 6000, 7000,…) to minimize the risk of overfitting to our training images.

### Convert model to allow real-time video processing

Thinking in a “real life, production” mindset, I would like to convert this model to run in real-time on a video feed. A model like this is meant to be used in a robotics type application, which brings about many important considerations - one of which is potentially limited computation power available. While YOLO_v2 can run at 40+ FPS, this requires a powerful GPU. Special attention will be required while training to work within the confines of available resources.



# Improving Data Collection

Web scraping can help to automate the process of collecting large amount of images for different objects we want to detect. Sites like Google (or any other service which supports natural language search or tagged images) can easily be scraped or offer data through APIs. Image relevance and quality can be QA’d during annotation.

There are many image annotation tools to label data with (scripts may need to be used to format the output correctly). Using a good annotation program can vastly improve the speed  of annotating images.

If budget is available, Amazon Mechanical Turk is a good choice to quickly build an annotated data set which can be validated by multiple annotators. There are both image and video annotation software which can be integrated with Mechanical Turk to improve the efficiency of the people completing the task.
