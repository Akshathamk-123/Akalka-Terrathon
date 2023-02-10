Collect the dataset and save it a file called 'data'. Create two folders 'train'(for training the model) and 'val'(for testing the model). Pick 20 images and save it in 
val folder, remaining to the train folder.

importing requirements: tensorflow, keras(from tensorflow), ImageDataGenerator(from tensorflow.keras.preprocessing.image) Conv2D(from tensorflow.keras.layers)

Loading and preprocessing the data: 
    training the data present in the 'train' and 'val' provide the path of the drive provide the path of both the dataset folders.

Building the model: 
    Using the sequential CNN model activation function: relu ,applies the rectified linear unit activation function. With default values, this returns the standard ReLU 
                                                        activation: max(x, 0), the element-wise maximum of 0 and the input tensor 
    MaxPooling2D: Downsamples the input along its spatial dimensions (height and width) by taking the maximum value over an input window (of size defined by pool_size ) 
                  for each channel of the input 
  
Compiling the model: 
    categorical_crossentropy: Used as a loss function for multi-class classification model where there are two or more output labels 
    optimizer as 'adam' : Adam optimizer is the extended version of stochastic gradient descent(SGD), optimization algorithm to find the model parameters that correspond to 
                          the best fit between predicted and actual outputs.
    metrics used is 'accuracy'

Training the model: 
      steps_per_epoch is chosen to be 100, which is quotient of total training samples by batch size chosen 
      epochs is chosen to be 50, once all images are processed one time individually of forward and backward to the network, then that is one epoch 
      validation steps is chosen to be 50

Save the model: 
    save with .h5 extension which is a file format to store structured data
