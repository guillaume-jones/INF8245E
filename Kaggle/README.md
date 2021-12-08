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

Notes:
- Forgot to write specifics before VGG
- Hypertuners can be reloaded for more info on accuracy and hyperparameters

| #  | Architecture  | Save # | Specifics                                                                         | LR   | L2   | Dropout | Augmented | Best acc. | Submitted |
|----|---------------|--------|-----------------------------------------------------------------------------------|------|------|---------|-----------|-----------|-----------|
| 2  | AlexNet       |        | Extreme overfitting                                                               |      |      |         | No        | ~0.4      | No        |
| 3  | GoogLeNet     | 1      |                                                                                   |      |      |         | No        | 0.66      | Yes       |
| 3  | GoogLeNet     | 2      | Added extra dropout                                                               |      |      |         | No        | 0.66      | Yes       |
| 3  | GoogLeNet     | HT     | Tried different L2 (best: 0.0025) and extra inception layers (worse)              |      |      |         | No        | 0.59      | No        |
| 3  | GoogLeNet     | 3      | Augmented data and batch normalization, then fine-tuning. 220 epochs              | 1E-4 | 2.5E-3 | 0.75  | Yes       | 0.71      | No        |
| 4  | Custom Resnet |        | Tried different types of regularization, always overfit                           |      |      |         | Yes       | ~0.5      | No        |
| 5  | ResNet        |        | Created from existing architecture, fast learning                                 |      |      |         | No        | ~0.5      | No        |
| 6  | VGG           | 0      | 60 epochs                                                                         | 1E-3 | 0    | No      | Yes       | 0.5       | No        |
| 6  | VGG           | 1      | 150 epochs                                                                        | 1E-3 | 0    | 0.3     | Yes       | 0.65      | No        |
| 6  | VGG           | 2      | Continue training from save 1 with learn_rate=0.0001                              | 1E-4 | 0    | 0.3     | Yes       | 0.69      | Yes       |
| 6  | VGG           | HT     | Hypertuning with different levels of dropout and L2                               | 1E-3 | -    | -       | Yes       | 0.65      | No        |
| 6  | VGG           | 3      | Continue training 2nd best HT model                                               | 1E-4 | 1E-5 | 0.5     | Yes       | 0.73      | No        |
| 6  | VGG           | 4      | 150 epochs                                                                        | 1E-3 | 1E-4 | 0.4     | Yes       | 0.68      | No        |
| 6  | VGG           | 5      | Fine tune save 3 with 5 epochs of normal data                                     | 1E-5 | 1E-5 | 0.5     | No        | 0.76      | No        |
| 6  | VGG           | 6      | Fine tune save 2 with 5 epochs of normal data                                     | 1E-4 | 1E-5 | 0.5     | No        | 0.80      | Yes       |
| 7  | DeeperVGG     | 1      | 150 epochs                                                                        | 1E-3 | 1E-5 | 0.3     | Yes       | 0.67      | No        |
| 7  | DeeperVGG     | 2      | Fine tune save 0 with 5 epochs of normal data                                     | 1E-4 | 1E-5 | 0.3     | No        | 0.77      | Yes       |
| 7  | DeeperVGG     | 3      | Tried SpatialDropout                                                              | 1E-3 | 1E-4 | 0.3     | Yes       | 0.65      | No        |
| 7  | DeeperVGG     | 4      | Fine-tuned save 3 with augmented and normal data                                  | 1E-4 | 1E-4 | 0.3     | Yes       | 0.79      | Yes       |
| 7  | DeeperVGG     | 5      | Tried minor changes and final layer of 64 + fine-tuning                           | 1E-4 | 1E-4 | 0.3     | Yes       | 0.78      | No        |
| 7  | DeeperVGG     | HT     | Add new 1024-deep layer + more batch norm + extra dropout + conv L2               | 5E-4 | -    | -       | Yes       | 0.78      | No        |
| 7  | DeeperVGG     | 6      | Fine tune best HT model with non-augmented data                                   | 1E-4 | 1E-3 | 0.4     | Yes       | 0.84      | No        |
| 7  | DeeperVGG     | 7      | Train save 6 for 4 epochs with validation data                                    | 1E-4 | 1E-3 | 0.4     | No        | ~0.85     | No        |
| 8  | VGG Res       | HT     | Hypertuning learning rate, dropout and L2                                         | 1E-3 | -    | -       | Yes       | 0.65      | No        |
| 8  | VGG Res       | 1      | 150 epochs + learning rate decrease + fine-tuning                                 | 1E-3 | 1E-3 | 0.5     | Yes       | 0.76      | No        |
| 8  | VGG Res       | 2      | 150 epochs + learning rate decrease                                               | 5E-4 | 1E-4 | 0.6     | Yes       | 0.70      | No        |
| 8  | VGG Res       | 3      | Fine-tuning on save 2                                                             | 5E-4 | 1E-4 | 0.56    | No        | 0.79      | No        |
| 9  | DeeperVGG2    | 1      | Tried 50% deeper initial layers 220 epochs + learning rate decrease               | 5E-4 | 1E-2 | 0.4     | Yes       | 0.79      | No        |
| 9  | DeeperVGG2    | 2      | Fine-tuning on save 2                                                             | 1E-4 | 1E-2 | 0.4     | No        | 0.84      | No        |
| 9  | DeeperVGG2    | 3      | Train save 2 for 4 epochs with validation data                                    | 1E-4 | 1E-2 | 0.4     | No        | ~0.85     | No        |
| 10 | WideResNet    | 0      | First WideResNet, train for 50 epochs (9 hours!), wildly variable performance     | 1E-3 | 1E-4 | 0.4     | Yes       | ~0.60     | No        |
| 10 | WideResNet    | 1      | Shrink WideResNet to n=2, k=10-12, with low regularization                        | 5E-4 | 1E-4 | 0.3     | Yes       | 0.67      | No        |
| 10 | WideResNet    | 2      | Train model with n=2, k=8                                                         | 5E-4 | 1E-4 | 0.4     | Yes       | 0.73      | No        |
| 10 | WideResNet    | 3      | Fine tune save 2 with non-augmented data                                          | 1E-5 | 1E-4 | 0.4     | No        | 0.80      | No        |
| 10 | WideResNet    | HT1    | Train 2 models with k=8, n=1 and drop=0.6 and n=2 and drop=0.3 (best)             | 5E-4 | 1E-5 | -       | Yes       | 0.77      | No        |
| 10 | WideResNet    | 4      | Fine tune best model from HT1 with non-augmented data                             | 1E-5 | 1E-4 | 0.3     | No        | 0.81      | No        |
| 10 | WideResNet    | HT2    | Train 3 models with n=2, dropout 0.3 or 0.5 and k=6 or 9 (best 0.5 and 9)         | 5E-4 | 1E-4 | -       | Yes       | 0.80      | No        |
| 10 | WideResNet    | 5      | Fine tune best model from HT2 with non-augmented data                             | 1E-5 | 1E-4 | 0.4     | No        | 0.84      | No        |
| 10 | WideResNet    | 6      | Try n=2, k=12, SpatialDropout2D and higher L2                                     | 5E-4 | 2E-4 | 0.5     | Yes       | 0.71      | No        |
| 10 | WideResNet    | HT3    | Deeper with n=3 and k=12. Tried different L2/dropout. Underfit and overfit        | 1E-4 | -    | -       | Yes       | 0.79      | No        |
| 10 | WideResNet    | HT3    | Deeper with n=3 and k=12. Tried different L2/dropout. Underfit and overfit        | 1E-4 | -    | -       | Yes       | 0.79      | No        |
| 10 | WideResNet    | 8      | Fine tuned best model from HT3 (l2=0.001, drop=0.5)                               | 1E-5 | 1E-3 | 0.5     | No        | 0.84      | No        |
| 11 | Stacking      | HT     | Try different dropouts and L2 regs for stacking. Overfit (lost 1.5% on test data) | 5E-5 | -    | -       | No        | 0.86      | Yes       |
| 11 | Stacking      | 0      | Try 2 Dense layers, 512 then 256.                                                 | 1E-5 | 1E-4 | 0.3     | No        | 0.86      | No        |
| 11 | Stacking      | 1      | Add SimpleNets, 3x256 layers                                                      | 1E-4 | 1E-1 | 0       | No        | 0.89      | No        |
| 11 | Stacking      | 2      | Add more SimpleNets                                                               | 1E-4 | 1E-1 | 0       | No        | 0.89      | Yes       |
| X  | Basic stack   | 1      | Stacked best VGG, 2 good DeeperVGGs and VGG Res. Overfit to valid set by 1%       | -    | -    | -       | No        | 0.85      | Yes       |
| X  | Basic stack   | 2      | Added DeeperVGG2_2                                                                | -    | -    | -       | No        | 0.86      | Yes       |
| X  | Basic stack   | 3      | Try stacking DVGG_7 and DVGG2_3 (train + valid). Improved performance 0.5%        | -    | -    | -       | No        | 0.87      | Yes       |
| X  | Basic stack   | 4      | Added WideResNet_4 to list from Basic stack 2.                                    | -    | -    | -       | No        | 0.86      | Yes       |
| X  | Basic stack   | 5      | Added WideResNet_5 to list from Basic stack 4.                                    | -    | -    | -       | No        | 0.87      | Yes       |
| X  | Basic stack   | 6      | Tweak prob and accuracy rating to 1 and 13                                        | -    | -    | -       | No        | 0.87      | Yes       |
| X  | Basic stack   | 7      | Add WideResNet_8 and use 1 and 7 prob/accuracy                                    | -    | -    | -       | No        | 0.87      | Yes       |
| X  | Basic stack   | 8      | Repeat save 7 with prob/accuracy of 4 and 8. Worse results! (avoid prob>1)        | -    | -    | -       | No        | 0.86      | Yes       |
| X  | Basic stack   | 9      | Add SimpleNet models to basic stack. High acc works best                          | -    | -    | -       | No        | 0.90      | Yes       |

Basic stacking :
- Altered data to give more weight to models with higher accuracy and confidence. Adjustable accuracy/probability weightings

### Notes

General
- Since we are doing multi-class classification, we should always use sparse cross-entropy loss for our models.
- To predict correctly, should use either from_logits=True in loss function or activation='softmax' in last layer
- Overfitting evidence : training accuracy gets ahead of valid accuracy
- Underfitting evidence : training accuracy is always equal to valid accuracy, and remains low
- Use BayesianTuner if you have Float values, and RandomChoiceTuner for Choice values
- Once we have obtained our best model, we should train it on all the training data

Architecture
- [Review of CNN architectures for image classification](https://machinelearningmastery.com/review-of-architectural-innovations-for-convolutional-neural-networks-for-image-classification/)
- Dropout layers are typically used with rate=0.5 after Dense layers and rate=0.1 after convolutional layers
- RandomZoom seems to break certain models (no learning at all)

Learning Rate
- [Learning rate blog post](https://www.jeremyjordan.me/nn-learning-rate/)
- Lowering the learning rate when valid accuracy stops increasing can give 2-5% more accuracy
- Fine-tuning with 5-10 epochs of non-augmented data can boost accuracy 5-10%

GoogLeNet
- [Original paper](https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf)
- Architecture: [table of parameters](https://media.geeksforgeeks.org/wp-content/uploads/20200429201421/Inception-layer-by-layer.PNG), [diagram](https://miro.medium.com/max/5176/1*ZFPOSAted10TPd3hBQU8iQ.png)
- [Implementation with Keras](https://machinelearningmastery.com/how-to-implement-major-architecture-innovations-for-convolutional-neural-networks/)

ResNet
- [Original paper, including architecture](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)
- [Implementation with Keras](https://machinelearningmastery.com/how-to-implement-major-architecture-innovations-for-convolutional-neural-networks/)

Wide ResNet
- [Original paper](https://arxiv.org/pdf/1605.07146.pdf)
- [Detail of paper](https://modelzoo.co/model/wide-residual-networks)
- [PyTorch implementation, consider trying in Keras](https://brandonmorris.dev/2018/06/30/wide-resnet-pytorch/)
- [Keras implementation](https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/applications/wide_resnet.py), different Batch-Relu-Conv order and missing ReLU for 1x1 conv blocks of shortcuts
 
Vision Transformer
- Transformers apparently need a lot of data (consider strong data augmentation)
- [Keras example](https://keras.io/examples/vision/image_classification_with_vision_transformer/)

Swin Transformer
-Apparently needs a lot of data like the vision transformer
-[Keras example](https://keras.io/examples/vision/swin_transformers/)

Dataset is called Animals-10
- Very little info/usage
- Several example on [Kaggle page](https://www.kaggle.com/alessiocorrado99/animals10/code) but they are mostly transfer learning
