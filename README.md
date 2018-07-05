# CNN-with-FashionMNIST
This is a repository for my first PyTorch Neural Network trained on FashionMNIST dataset.

## The FashionMNIST dataset
This dataset can be found [here](https://github.com/zalandoresearch/fashion-mnist). It consists of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes.
This dataset is harder than the MNIST dataset and hence accuracy on this dataset with same architecture will be less than that on the MNIST dataset.

## The Dataset Class
PyTorch provides a way to create custom dataset classes and then load them during training as testing using data loader. The FashionMNIST dataset files are ubyte files which are in `.gzip` format. One can use either gzip python module or gzip manually and then use normal python file operations to open the files. The first 16 hexadecimal digits contain information like length,number of rows,number of columns,etc. The data after that contains the images. The training and test images sets are reshaped into tensors of shape **(60000,28,28)** and **(10000,28,28)** while the labels sets are converted to one-hot encoded tensors in the shape **(60000,10)** and **(10000,28,28)** respectively. 
 
## The Model
Some advantages and shortcomings of my model are discussed below -
- The model uses a new activation function called Swish activation. It has been seen to perform better than ELU and ReLU in some cases and thus, it has been implemented in the code written in PyTorch and used in all the 3 Convolutional Layers.
- The optimizer is Adagrad, which performs similar to the Adam. On this dataset, SGD with Nesterov Momentum gives a higher performance than Adagrad.
- There is a learning rate decay at every 10 steps by a factor of 0.1 to improve the performance.
- The model is trained using BCE Loss over a Softmax Output because CrossEntropyLoss between 2 tensors in PyTorch cannot be calculated directly. BCE Loss in PyTorch is unstable and therefore other choices can be used.

- The parameter initialization is Xavier Normal
- The layers in sequence are:

  - Convolutional layer with 16 feature maps of size 3 x 3
  - BatchNorm layer followed by Swish activation.
  - Max Pooling layer of size 2 x 2.
  - Convolutional layer with 32 feature maps of size 3 x 3
  - BatchNorm layer followed by Swish activation.
  - Max Pooling layer of size 2 x 2.
  - Convolutional layer with 64 feature maps of size 3 x 3
  - BatchNorm layer followed by Swish activation.
  - Max Pooling layer of size 2 x 2.
  - Fully connected layer of size 10.
  - Softmax Layer of size 10.

## Accuracy Details
- After training for 18000 iterations on a minibatch size of 100, using initial learning rate 0.015, the accuracy achieved using this model is 99.2% on MNIST and 92.1% on FashionMNIST dataset.

- The model can be tweaked a bit to give similar or higher accuracy with lesser (2) layers, if we use SGD with Nesterov Momentum and use deeper initial layers, say 32 instead of 16 and so on. This gives an accuracy of about 93.11 % on FashionMNIST and 99.6% on MNIST.

## Other Details
- The data has not been preprocessed in any manner. One could preprocess and add transforms like RandomHorizontalFlip() or cropping and rescaling and so on, to get even better performance on the test dataset, to avoid overfitting.

- The training was done on Google Colab using their GPU for faster results.

## Running The Script
- The script can be run using `python3 fashionmnist.py`
