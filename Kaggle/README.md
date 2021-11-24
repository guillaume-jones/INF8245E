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

Tensorflow Keras:
- [Keras quickstart](https://www.tensorflow.org/tutorials/quickstart/beginner)
- [Keras classification tutorial](https://www.tensorflow.org/tutorials/keras/classification)
- [Keras Convnets tutorial](https://www.tensorflow.org/tutorials/images/cnn)
- [Keras Dataset tutorial](https://www.tensorflow.org/guide/data#batching_dataset_elements)
- [Keras reference](https://keras.io/api/)
- [Keras Layers reference](https://keras.io/api/layers/)
- [How to use Keras with a GPU](https://www.tensorflow.org/guide/gpu)

Keras Tuner:
- [Keras Tuner reference](https://keras.io/api/keras_tuner/)
- [Keras Tuner tutorial](https://www.tensorflow.org/tutorials/keras/keras_tuner)
- [Keras Tuner tutorial 2](https://neptune.ai/blog/keras-tuner-tuning-hyperparameters-deep-learning-model)

### Notes

- Since we are doing multi-class classification, we should always use sparse cross-entropy loss for our models.
- To predict correctly, should use either from_logits=True in loss function or activation='softmax' in last layer
