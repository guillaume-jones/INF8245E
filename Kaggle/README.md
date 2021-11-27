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

Use F1-Micro score (same as accuracy for multiclass)

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

### Models so far

| Model # | Architecture  | Save # | Specifics                                                                                | Augmented | Best accuracy | Submitted |
|---------|---------------|--------|------------------------------------------------------------------------------------------|-----------|---------------|-----------|
| 2       | AlexNet       | -      | Extreme overfitting                                                                      |           | ~0.4          | No        |
| 3       | GoogLeNet     | 1      |                                                                                          | No        | 0.66          | Yes       |
| 3       | GoogLeNet     | 2      | Different regularization                                                                 | No        | 0.63          | No        |
| 3       | GoogLeNet     | 3      | Added extra dropout                                                                      | No        | 0.66          | Yes       |
| 3       | GoogLeNet     | HT     | Tried different L2d and extra inception layers. Best L2 is 0.0025. Extra layers = worse. | No        | 0.59          | No        |
| 4       | Custom Resnet | -      | Tried different types of regularization                                                  | Yes       | ~0.5          | No        |
| 5       | ResNet        | -      | Created from existing architecture, fast learning, had to modify data for 3 channels     | No        | ~0.5          | No        |
| 6       | VGG           | 0      | No dropout or regularization, 60 epochs                                                  | Yes       | 0.5           | No        |
| 6       | VGG           | 1      | Added dropout=0.3, no regularization, learn_rate=0.001, 150 epochs                       | Yes       | 0.65          | No        |
| 6       | VGG           | 2      | Continue training save 1 with learn_rate=0.0001 until val_loss stopped decreasing        | Yes       | 0.69          | Yes       |
| 6       | VGG           | HT     | Hypertuning with different levels of dropout and L2. Reload HT for details               | Yes       | 0.65          | No        |
| 6       | VGG           | 3      | Continue training HT model #2 (dropout=0.5, reg=0.00001) with learn_rate=0.0001          | Yes       | 0.73          | No        |
| 6       | VGG           | 4      | Dropout=0.4, l2_reg=0.0001. Trained first with lr=0.001, then lr=0.0001                  | Yes       | 0.68          | No        |

### Notes

General
- Since we are doing multi-class classification, we should always use sparse cross-entropy loss for our models.
- To predict correctly, should use either from_logits=True in loss function or activation='softmax' in last layer
- Overfitting evidence : training accuracy gets ahead of valid accuracy
- Underfitting evidence : training accuracy is always equal to valid accuracy, and remains low
- Use BayesianTuner if you have Float values, and RandomChoiceTuner for Choice values

Architecture
- Dropout layers are typically used with rate=0.5 after Dense layers and rate=0.1 after convolutional layers
- RandomRotation does not seem to work very well, maybe because it creates 0-pixels
- [Review of CNN architectures for image classification](https://machinelearningmastery.com/review-of-architectural-innovations-for-convolutional-neural-networks-for-image-classification/)

Learning Rate
- [Learning rate blog post](https://www.jeremyjordan.me/nn-learning-rate/)

GoogLeNet
- [Main Paper](https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf)
- Architecture: [table of parameters](https://media.geeksforgeeks.org/wp-content/uploads/20200429201421/Inception-layer-by-layer.PNG), [diagram] (https://miro.medium.com/max/5176/1*ZFPOSAted10TPd3hBQU8iQ.png)
- [Implementation with Keras](https://machinelearningmastery.com/how-to-implement-major-architecture-innovations-for-convolutional-neural-networks/)

ResNet
- [Main Paper, including architecture](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)
- [Implementation with Keras](https://machinelearningmastery.com/how-to-implement-major-architecture-innovations-for-convolutional-neural-networks/)

Dataset is called Animals-10
- Very little info/usage
- Several example on [Kaggle page](https://www.kaggle.com/alessiocorrado99/animals10/code) but they are mostly transfer learning
