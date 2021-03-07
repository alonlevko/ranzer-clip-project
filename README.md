In this readme we will explain what is each file and were some of the code was taken from.

All the files that have cpgan in their name were cloned from https://github.com/basilevh/object-discovery-cp-gan.
We changed the file cpgan_model.py to include also the class MyMiniUnet.
We changed the file cpgan_data.py so that the transforms will be what we need for the x-rays, 
and we also changed the way images are loaded a little bit so that loading grayscale images 
won't be a problem.
We changed train_cpgan.py so that if there is not saved model in memory it will load the 
pre trained Unet as it's generator.

The file create_masks.py is a script we wrote for creating the masks and is adapted from a 
kaggle notebook on visualizing the catheter lines.

The file Gneerate_augmented_dataset.py generates the augmented dataset using the Unet model.
In this file, in comments there is the part for generating the masks agumented dataset.

resize_check.py is a script we wrote and is used to load an image and generate it in different sizes so we could pick the right size.

split_train_val.py is a script we wrote for generating the train, validation and masks for the 
foreground / background images needed for the cpgan.

Unet_pre_train.py is the script we wrote and used for pre training our Unet for segmenting the lines.

Unet_Prediction.py is a script we wrote and used for visualizing segmentation results of the Unet.

experiment.py is the script we wrote and used for the last part of out workflow, were it splits the test-set, trains
models on the 3 datasets and checks the score for each model on the test dataset. 
