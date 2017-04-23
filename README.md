# **Behavioral Cloning**
## Utilizing Deep Convolutional Neural Networks to Clone Driving Behaviors
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[//]: # (Image References)

[image1]: ./Pics/cover.gif
[image2]: ./Pics/preprocess.png
[image3]: ./Pics/mirror.png
[image4]: ./Pics/transformation.png
[image5]: ./Pics/translation.png
[image6]: ./Pics/original_dist.png
[image7]: ./Pics/augmented_data.png
[image8]: ./Pics/nvidia.png
[image9]: ./Pics/model.png
[image10]: ./Pics/multi-cam.png
[image11]: ./Pics/steering_angles.png
[image12]: ./Pics/dataset.png


![alt text][image1]
## __Overview__
###### OBJECTIVE
Given the dash-cam of footage of a car we need to be able predict the steering angles in real-time to enable it to drive autonomously.

###### APPROACH
To achieve this we will be employing behavioral cloning. We start with a simulator built with Unity. We will drive through the track and collect data through the 3 onboard dash-cams, to record how a human navigates the track. This data will used to train our network. After which it will run in real-time to allow the vehicle to navigate the track autonomously.

###### ISSUES
This is a harder task to perfect than a classification problem. Due to these 2 reasons:

* There is no accurate quantitative metric to analyze the performance of the model. A model may have an extremely low validation loss but it may not translate into great performance on the track or imply smooth driving. This is due to the fact that this model does not take into account temporal information like an RNN, so it does not take into account past steering angles. So most of the time, you would have to rely on qualitative observations on which parts of the track the model is not performing well and then proceed to improve your model/data accordingly.

* Training Data has to be somewhat curated and intentionally generated, because it does not contain recovery driving. For example, if the human driver drove around the track while being centered in the middle of the road, the car would only know how to drive when its centered in the middle of the road. It would not know how to steer back to the center of the road when its along the road edges. So the original data only teaches it how to drive perfectly, it does not teach recovery driving. In the documentation below I will elaborate on the steps I took to fix this issue.


##  __Solution Design Approach__

The overall strategy for deriving a model architecture was to start with a very simple model architecture and very small amount of data and try to achieve overfitting. If I could achieve overfitting with good test accuracy it would mean my data could be represented well with my model. This also allowed me to conduct many experiments on data augmentation and preprocessing very quickly and converge onto what worked.

## __Preprocessing__

![alt text][image2]

###### NORMALIZATION
We normalize the input to minimize sensitivity and make the network more stable. Experimented with normalizing to [0,1] with mean shift of -0.5 but this delivered more favourable results.
```python
model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=input_shape))
```


###### COLOR SPACE
I also decided to let the model learn what colorspace is best instead of manually specifying which colorspace to use. Previously experimented with sliced HSV colorspace followed by gaussian blur of kernel_size 5.

```python
model.add(Conv2D(3, kernel_size=(1, 1), strides=(1, 1), activation='linear'))
```

###### CROP
Cropped the image, to retain only the information relevant to steering angle decision making in that moment. Explored more aggressive cropping after reading David Ventimiglia in his [post](http://davidaventimiglia.com/carnd_behavioral_cloning_part1.html?fb_comment_id=1429370707086975_1432730663417646&comment_id=1432702413420471&reply_comment_id=1432730663417646#f2752653e047148) which he mentioned, "For instance, if you have a neural network with no memory or anticipatory functions, you might downplay the importance of features within your data that contain information about the future as opposed to features that contain information about the present."

```python
model.add(Cropping2D(cropping=((80, 25), (0, 0))))
```

These processes are part of the keras model, so they are not only applied on the training data but also to the dash-cam images in real-time while in autonomous mode.


## __Expanding Model Architecture__

The basic model was able to navigate the track successfully, but the driving was not ideal as it would stray too far off center at times. So after reading the research paper by Nvidia on End to End learning for self-driving cars. I implemented their model with some slight tweaks to achieve a smoother driving experience.

| Modified       | Nvidia           |
| ------------- |:-------------:|
| ![alt text][image9]  | ![alt text][image8] |

My model was showing signs of overfitting as the mean squared error on the training set(80% of data) would drop lower than that of the validation set(20% of data).To combat the overfitting, I modified by increasing the dropout to 0.2, which increased its performance and stability.

The Adam optimizer was chosen with default parameters and the chosen loss function was mean squared error (MSE).

The next step was to run the simulator to see how well the car was driving around the track. The vehicle was able to stay on the road the entire time but was still underperforming at some corners and drifting too far out. At this point I had to use my intuition to augment the data in such a way that it could fix the behaviour of the car.

###### TRAINING
Model was trained locally on a Macbook with a GPU. It took about 5 Epochs to get a successful model.

## __Data Augmentation__

There are a few ways in which we can improve the performance of the car, one of which would be to simply generate more test data for specific cases by driving on the track. But I wanted to just work with the original dataset given by Udacity, as I wanted to see how well I could work with what I have and augment the dataset to develop a better model. So I choose the path of data augmentation.

###### ORIGINAL DRIVING DATA

In most of the recorded instances of the data, the car is centered in the lane and it's center of vision is angled towards its desired location. Unlike recovery driving where your car's center of vision is not angled towards your future destination. As such the model trained just on the original dataset did not perform exceptionally when recovering from the edges of the road.

![alt text][image12]

The amount of each augmentation that is present in the data set was tuned based on how the car performed on the track. This was done using a random generator with a normal distribution coupled with a limit that I could change to specify how much of the data I want augmented with a certain function.

Augmentation functions were referenced from this [blog post ](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9) by Vivek Yadav.

###### MULTI CAMERA
![alt text][image10]

The car has 3 different cameras, each with a different perspective. For each lap we would obtain 3 streams of data. We can use these left and the right cameras to simulate the car being off-centered in the road. For each of these images we can then propose a value for which it should steer to get back to the center of the lane. Too high a value and the car would be swerving through the track and if too low it would not be able to compensate enough to center itself in the lane. The amount of side-camera examples used in the data set was limited by a control value, as including all the examples resulted in the car constantly swerving from side to side and too little of these data and it would not know how to recover well from being off center.


![alt text][image3]

As left turning bends are more prevalent than right bends in the training track, we use this augmentation to balance out the dataset and help increase the generalization of our model.


![alt text][image5]

Translates the image left and right to simulate the car being off center from the road. After augmentation the center of vision of the car is no longer it's desired future destination. The amount of translation is scaled and summed to the steering value to encourage the car to compensate of the off-center behaviour. This also helps the car realign itself if it stray's from the desired path.


![alt text][image4]

To simulate more extreme turns and provide more examples for higher turning angles, horizontal affine transformations were applied to the images. The amount by which it was transformed was scaled and summed to the steering angle. Similarly we can notice that the center of vision of the car is no longer it's desired future destination.

## __Data Visualized__


As seen from the data plot below the steering angles are  heavily skewed toward low and zero turning angles. This is due to the nature of the track as it includes long sections with very slight or no curvature. This is not ideal for the neural network as it becomes heavily biased towards driving straight or at low angles and may by unable to handle sharp corners.

It is also notable that the steering angles jump from value to value that may cause jerky turns as compared to a smooth and consistent turn.

###### _Normal Scale and Log Scale Graphs_
![alt text][image6]

After augmenting the data we now have a more balanced dataset with more examples for higher turning values and intermediate turning values. This results in a more dense representation of the spectrum of turning values which helped in enabling smoother transitions between steering angles while driving.

###### _Normal Scale and Log Graphs_
![alt text][image7]

## __Conclusion__

Aggressive dropout combined with controlled data augmentation resulted in the best performing model that enabled the vehicle to drive autonomously around the track in a smooth manner.

###### DATA
Understanding the original data distribution was essential. However understanding how to change the distribution to fix certain driving situation failures was crucial.

As images were obtained from a video stream it would be beneficial to remove very similar images and steering values to increase the quality of the data, especially during the long straight portions of the track.

###### AUGMENTATION
Applying augmentations evenly through the data yielded good results. However in order to obtain desired behaviors we can control how much of the data is augmented by a certain function. As an abundance of some of the generated data actually had a negative impact on the network. We also control how much of the overall data is augmented. This step helped to stabilize the model greatly.

###### QUANTITATIVE METRIC
Would like to try and experiment with steering value comparisons. To compare the predicted steering value with the averaged steering value through the track. This would be a good metric to represent how much of the drivers behaviour is actually cloned.

###### *Original Driver's Steering Angles*
![alt text][image11]

###### FURTHER IMPROVEMENTS
For a more robust solution one should consider including temporal information as part of the model as driving is not just about instantaneous decisions based solely on the scene, it requires context and past steering values to normalize the behaviours.

Currently one dataset provides both the steering angles for scenes and also the driving logic. Humans learn the steering angles through visual recognition and intuition, but our driving behaviour is shaped by a set of rules and expectations that are pre-defined. It would be interesting to see how we can automate the learning of such driving rules, to create a driving model that is more familiar to humans.

Another suggestion would be more dash-cams at different parts of the car (which are trained to recognize different type of scene's) to increase the confidence of steering predictions.

Reinforcement Learning!!! Using a fitness function that rewards the car for staying centered on the road and reduces in reward the further away you drift from the center.


## Details About Files In This Directory

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument `run1` is the directory to save the images seen by the agent to. If the directory already exists it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp when the image image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Create a video based on images found in the `run1` directory. The name of the video will be name of the directory following by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

The video will run at 48 FPS. The default FPS is 60.

#### Why create a video

1. It's been noted the simulator might perform differently based on the hardware. So if your model drives succesfully on your machine it might not on another machine (your reviewer). Saving a video is a solid backup in case this happens.
2. You could slightly alter the code in `drive.py` and/or `video.py` to create a video of what your model sees after the image is processed (may be helpful for debugging).
