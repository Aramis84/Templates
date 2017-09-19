from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
import collections

input_size = (64, 64,3) 
num_classes = 2

num_training_samples = 8000
num_test_samples = 2000

def makeModel(conv_layers = 3, fully_connected_layers = [64, 64], input_dims = (32,32,3), pool = 2, p = 0.5, \
              filter_stride = 1, pool_stride = 2, pad =  'same', seed = 10, output_dim = 10, \
              use_batchnorm = False, solver = 'adam', metric = 'accuracy', regularizer = regularizers.l2(0.01), \
              filter_dims = 3, filter_num = 32):
    
    """
    Creates a model by stacking convolutional and fully connected layers
    Inputs : conv_layers -> number of conv + pooling layers
             fully_connected_layers -> list with number of units in the fully connected hidden layers (excluding the output layer)
             input_dims -> 3D tensor for the input size of image
             pool -> size of pooling filter (height and width are assumed same)
             p -> dropout rate (dropout is applied only on the fully connected layers)
             filter_stride -> stride used for filter kernel
             pool_stride ->  stride used for pooling kernel
             pad -> padding used before applying filter kernel 'same' so that ouptut size is same as input, 'valid' means no padding,
             'casual' for dilated convolutions
             seed-> random seed (may be useful for gradient checking)
             output_dim -> number of classes in the classification task
             use_batchnorm -> Boolean whether to use spatial batch normalization in the convolution layers
             solver -> parameter update rule
             metric -> performance evaluation criterion
             regularizer -> regularization (default is L2 regularization with parameter 0.01)
             filter_dims -> size of filter kernel (height and width assumed to be same)
             filter_num -> list of number of filters to use for each successive layers. If one number, then same number of filters are used 
             for each layer
     
    Outputs : classifier model
    """        
       
    classifier = Sequential()
    
    num_conv = conv_layers
    num_hidd = len(fully_connected_layers)
    f_dims = [0,0]
    if isinstance(filter_dims, collections.Iterable):
        f_dims[0] = filter_dims[0]
        f_dims[1] = filter_dims[1]      
    else:
        # height and width are same
        f_dims[0] = f_dims[1] = filter_dims
        
    
    if isinstance(filter_num, collections.Iterable):
        f_nums = list(filter_num)
    else:
        # same filter for all layers
        f_nums = [filter_num]*num_conv
        
   
    classifier.add(Convolution2D(f_nums[0],(f_dims[0],f_dims[1]), strides = filter_stride, padding = pad, \
                                 input_shape = input_dims, activation = 'relu', kernel_regularizer = regularizer))
    classifier.add(MaxPooling2D(pool_size = pool, strides = pool_stride))
    if use_batchnorm:
        classifier.add(BatchNormalization())
    
    for ii in range(num_conv-1):
        classifier.add(Convolution2D(f_nums[ii+1],(f_dims[0],f_dims[1]), strides = filter_stride, padding = pad, \
                                  activation = 'relu', kernel_regularizer = regularizer))
        classifier.add(MaxPooling2D(pool_size = pool, strides = pool_stride))
    
        if use_batchnorm:
            classifier.add(BatchNormalization())
    
    classifier.add(Flatten())
           
    for jj in range(num_hidd):
        units = fully_connected_layers[jj]
        classifier.add(Dense(units = units, activation = 'relu', kernel_regularizer = regularizer))
        if p > 0:
            classifier.add(Dropout(rate= p, seed = seed))
             
    if output_dim == 2:
        acti = 'sigmoid'
        loss = 'binary_crossentropy'
        units = 1
    elif output_dim > 2:
        acti = 'softmax'
        loss = 'categorical_crossentropy'
        units = output_dim
    
    classifier.add(Dense(units = units,  activation = acti))
    classifier.compile(optimizer = solver, loss = loss, metrics = [metric])
    
    return classifier

def classiferTrain(epochs = 20, batch_size  = 32):
    
    classifier = makeModel(input_dims = input_size, output_dim = num_classes)
    
    train_datagen = ImageDataGenerator(
                    rescale=1./255,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True)
    
    test_datagen = ImageDataGenerator(rescale=1./255)
   
    if num_classes == 2:
        class_mode = 'binary'
    elif num_classes > 2:
        class_mode = 'categorical'
    
    train_generator = train_datagen.flow_from_directory(
                      'dataset/training_set',
                      target_size = input_size[:-1],
                      class_mode = class_mode,
                      batch_size = batch_size
                      )
    
    test_generator = test_datagen.flow_from_directory(
                      'dataset/test_set',
                      target_size = input_size[:-1],
                      class_mode = class_mode,
                      batch_size = batch_size
                      )
      
    classifier.fit_generator(train_generator,
                             steps_per_epoch = num_training_samples,
                             epochs = epochs,
                             validation_data = test_generator,
                             validation_steps = num_test_samples)
    
    
def main():
   classiferTrain()

if __name__ == '__main__':
    main()     
                             