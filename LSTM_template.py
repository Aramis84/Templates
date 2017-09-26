from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout

def makeModel(input_dims, output_dims,  lstm_layers = [4,4], p = 0.0, solver = 'rmsprop', activation = 'tanh',\
	weight_intializer = 'glorot_uniform', task = 'regression', seed = 10):
    """
    Creates a LSTM model by stacking multiple layers
    Inputs : input_dims -> number of features in input dataset
             output_dims -> number of outputs
             lstm_layers -> list with number of units in each LSTM layer
             p -> dropout rate 
             seed-> random seed (may be useful for gradient checking)
             solver -> parameter update rule
             weight_intializer -> intiialization meethod for weight matrices
             task -> 'classification' or 'regression'
             activation -> activation function to use (input, forget, output gates still use sigmoid)
     
    Outputs : model
    """        
   
    model = Sequential()
    num_lstm_layers = len(lstm_layers)
    if num_lstm_layers > 1:
        return_sequences = [True]* (num_lstm_layers-1) + [False]
    else:
        return_sequences = False
         
    model.add(LSTM(units = lstm_layers[0], activation = activation, input_shape = (None,input_dims), return_sequences = return_sequences[0]))
    
    for ii in range(num_lstm_layers-1):
        model.add(LSTM(units = lstm_layers[ii+1], activation = activation, return_sequences = return_sequences[ii+1]))
        if p > 0.0:
            model.add(Dropout(rate= p, seed = seed))
            
            
    if task == 'regression':
        assert(output_dims ==1), 'For regression output dimension must be 1'

    if task == 'classification'and output_dims == 2:
       acti = 'sigmoid'
       loss = 'binary_crossentropy'
    elif task == 'classification' and output_dims > 2:
       acti = 'softmax'
       loss = 'categorical_crossentropy'
    elif task == 'regression':
       loss = 'mean_squared_error'
       acti = None
       
       
    model.add(Dense(units = output_dims, activation = acti))
    if p > 0.0:
        model.add(Dropout(rate= p, seed = seed))
        
    model.compile(optimizer = solver, loss = loss)
    return model


# For hyperparameter tuning
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
model = KerasRegressor(build_fn = makeModel)
# The input and output dimensions and the task type need to passed on to the parameter dictionary
parameters = {'batch_size' : [32], 'epochs' : [100, 200], 'input_dims' : [1], 'output_dims' :[1], 'task' : ['regression'],\
              'solver' : ['rmsprop', 'adam'], 'lstm_layers' : [[4,4], [8,8]]}
grid_search = GridSearchCV(estimator = model, param_grid = parameters, scoring = 'neg_mean_squared_error')
