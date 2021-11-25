### Extra Instructions

Class labels : 
0: big_cats, 
1: butterfly, 
2: cat, 
3: chicken,
4: cow, 
5: dog, 
6: elephant, 
7: goat, 
8: horse, 
9: spider, 
10: squirrel

Use F1-Micro score.

### TensorFlow

Tutorials:
- [Keras quickstart](https://www.tensorflow.org/tutorials/quickstart/beginner)
- [Keras classification tutorial](https://www.tensorflow.org/tutorials/keras/classification)
- [Keras Convnets tutorial](https://www.tensorflow.org/tutorials/images/cnn)
- [Keras Dataset tutorial](https://www.tensorflow.org/guide/data#batching_dataset_elements)

Keras:
- [Keras reference](https://keras.io/api/)
- [Keras reference 2](https://www.tensorflow.org/api_docs/python/tf/keras)
- [Keras Layers reference](https://keras.io/api/layers/)
- [How to use Keras with a GPU](https://www.tensorflow.org/guide/gpu)

Keras Tuner:
- [Keras Tuner reference](https://keras.io/api/keras_tuner/)
- [Keras Tuner tutorial](https://www.tensorflow.org/tutorials/keras/keras_tuner)
- [Keras Tuner tutorial 2](https://neptune.ai/blog/keras-tuner-tuning-hyperparameters-deep-learning-model)

### Notes
General
- Since we are doing multi-class classification, we should always use sparse cross-entropy loss for our models.
- To predict correctly, should use either from_logits=True in loss function or activation='softmax' in last layer
- Overfitting evidence : training accuracy gets ahead of valid accuracy
- Underfitting evidence : training accuracy is always equal to valid accuracy, and remains low

Architecture
- Dropout layers are typically used with rate=0.5 after Dense layers and rate=0.1 after convolutional layers
- RandomRotation does not seem to work very well, maybe because it creates 0-pixels
- [Review of CNN architectures for image classification](https://machinelearningmastery.com/review-of-architectural-innovations-for-convolutional-neural-networks-for-image-classification/)

GoogLeNet
- [Main Paper](https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf)
- Architecture: [table of parameters](https://media.geeksforgeeks.org/wp-content/uploads/20200429201421/Inception-layer-by-layer.PNG), [diagram] (https://miro.medium.com/max/5176/1*ZFPOSAted10TPd3hBQU8iQ.png)
- [Implementation with Keras](https://machinelearningmastery.com/how-to-implement-major-architecture-innovations-for-convolutional-neural-networks/)
