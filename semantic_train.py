import pickle
import numpy as np
from keras import models
from keras.models import Sequential 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import PorterStemmer
from keras.datasets import imdb
from keras import preprocessing
import numpy as np
from keras.models import Sequential 
from keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import RMSprop
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import CSVLogger

class TrainANN():  
    def __init__(self, dir_y, dir_X):
        self.X = pickle.load(dir_X)
        self.y = pickle.load(dir_y)

    def CNN1D(self, 
              max_features = 10000, 
              embed_dim = 10, 
              maxlen = 1000, 
              lr = 0.001, 
              epochs = 20, 
              batch_size = 100, 
              verbose = 1,
              dropout_rate = 0.5,
              plot = True,
              save_model = True):
        '''
        A class method to train CNN 1D.

        Parameters
        ----------
        max_features : int, optional
            Parameter. The default is 10000.
        embed_dim : int, optional
            Parameter. The default is 10.
        maxlen : int, optional
            Parameter. The default is 1000.
        lr : float, optional
            Parameter. The default is 0.001.
        epochs : int, optional
            Parameter. The default is 20.
        batch_size : int, optional
            Parameter. The default is 100.
        verbose : int, optional
            Parameter. The default is 1.
        dropout_rate : float, optional
            Parameter. The default is 0.5.
        plot : boolean, optional
            Parameter. The default is True.
        save_model : boolean, optional
            Parameter. The default is True.

        Returns
        -------
        None.

        '''
        
        model = Sequential()
        model.add(layers.Embedding(max_features, embed_dim, input_length=maxlen))
        model.add(layers.Conv1D(32, 7, activation='relu'))
        model.add(layers.MaxPooling1D(5))
        model.add(layers.Dropout(dropout_rate))
        model.add(layers.Conv1D(32, 7, activation='relu'))
        model.add(layers.GlobalMaxPooling1D())
        model.add(layers.Dropout(dropout_rate))
        model.add(layers.Dense(3))
        model.compile(optimizer=RMSprop(lr=lr), loss='categorical_crossentropy', metrics=['acc']) 
        
        print(model.summary()) 

        self.logCNN1D = CSVLogger('logCNN1D.txt', append=True, separator=';')           
        
        history = model.fit(self.train_X, self.train_y,
                            epochs = epochs,
                            batch_size = batch_size, 
                            validation_data = (self.val_X, self.val_y),
                            verbose = verbose,
                            callbacks = self.logCNN1D)
        
        self.history = history
                
        if plot:
            TrainANN.report(history,title="CNN")
        if save_model:
            model.save('CNN1D.h5')
            
    def SimpleRNN(self, 
                  max_features = 10000, 
                  embed_dim = 5, 
                  maxlen = 1000, 
                  lr = 0.001, 
                  epochs = 20, 
                  batch_size = 10, 
                  verbose = 1,
                  dropout_rate = 0.5,
                  plot = True,
                  save_model = True):
        '''
        A class method to train Simple RNN.

        Parameters
        ----------
        max_features : int, optional
            Parameter. The default is 10000.
        embed_dim : int, optional
            Parameter. The default is 10.
        maxlen : int, optional
            Parameter. The default is 1000.
        lr : float, optional
            Parameter. The default is 0.001.
        epochs : int, optional
            Parameter. The default is 20.
        batch_size : int, optional
            Parameter. The default is 100.
        verbose : int, optional
            Parameter. The default is 1.
        dropout_rate : float, optional
            Parameter. The default is 0.5.
        plot : boolean, optional
            Parameter. The default is True.
        save_model : boolean, optional
            Parameter. The default is True.

        Returns
        -------
        None.

        '''        
        
        model = Sequential() 
        model.add(layers.Embedding(max_features, embed_dim, input_length =  maxlen))
        model.add(layers.SimpleRNN(32))
        model.add(layers.Dropout(dropout_rate))
        model.add(layers.Dense(3, activation='sigmoid'))
        model.compile(optimizer = RMSprop(lr = lr), 
                      loss = 'categorical_crossentropy',
                      metrics = ['acc']) 
        model.summary()
        
        self.logSimpleRNN = CSVLogger('logSimpleRNN.txt', append=True, separator=';')           
        
        history = model.fit(self.train_X, self.train_y,
                            epochs = epochs,
                            batch_size = batch_size, 
                            validation_data = (self.val_X, self.val_y),
                            verbose = verbose,
                            callbacks = self.logSimpleRNN)
        
        self.history = history
        
        if plot:
            TrainANN.report(history,title="SimpleRNN")
        if save_model:
            model.save('SimpleRNN.h5')
            
    def evaluate_model(self, model_type):
        '''
        A class method to evaluate model

        Parameters
        ----------
        model_type : string
            Flag to tell whcih model to evaluate.

        Returns
        -------
        None.

        '''
        if model_type == 'SimpleRNN':
            model = models.load_model('SimpleRNN.h5')
            test = model.evaluate(self.test_X, self.test_y)
            print('loss:', test[0])
            print('accuracy:', test[1])
        elif model_type == 'CNN1D':
            model = models.load_model('CNN1D.h5')
            test = model.evaluate(self.test_X, self.test_y)
            print(model_type)
            print('loss:', test[0])
            print('accuracy:', test[1])
    
if __name__ == '__main__':
    TrainANN('sp500.txt', 'reddit_10')