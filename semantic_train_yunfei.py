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
from sklearn.preprocessing import OneHotEncoder as OHE
from keras.callbacks import CSVLogger
import pandas as pd
from datetime import datetime
from time import sleep
import re
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from text_data_processor import TextDataProcessor
from finance_data_collector import YahooData

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
from keras import models, layers
from keras import optimizers
from keras import regularizers
from keras.layers import SimpleRNN
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import roc_curve, auc, roc_auc_score

class TrainANN():  
    def __init__(self, dir_y, dir_X):
        file_X = open(dir_X, 'rb')
        self.X_var = pickle.load(file_X)
        file_y = open(dir_y, 'rb')
        self.y_var = pickle.load(file_y)
        
        self.date = pd.merge(pd.DataFrame(self.X_var.data.Date), pd.DataFrame(self.y_var.yf.index))
        
        self.X_df = pd.merge(self.X_var.data, self.date, on = 'Date')
        
        self.y_df = pd.merge(self.y_var.yf, self.date, on = 'Date')
        
        y = [[x] for x in self.y_df['Close_Sign']]
        
        self.y = OHE(sparse = False).fit_transform(y)

        
        
    def stem(str_input):
        '''
        A stand-alone function that stems a sentence.

        Parameters
        ----------
        str_input : str
            A string variable to be stemmed.

        Raises
        ------
        ValueError
            Raise ValueError when more than 1 group in token pattern is captured.

        Returns
        -------
        words : lst
            Post stemming list.

        '''
        
        token_pattern = re.compile(r"(?u)\b\w\w+\b")
    
        if token_pattern.groups > 1:
            raise ValueError(
                "More than 1 capturing group in token pattern. Only a single "
                "group should be captured."
            )
    
        words = token_pattern.findall(str_input)
        
        ps = PorterStemmer()
        
        words = [ps.stem(word) for word in words]
        
        return words
    
    def vectorize(self,
                  full_query = True, 
                  ana = 'word',
                  sw = 'english',
                  lc = True,
                  bi = False,
                  st = True,
                  rare_word = 100):
        '''
        Process the predictors to prepare for model training

        Parameters
        ----------
        full_query : BOOL, optional
            Include comment or not. The default is True.
        ana : str, optional
            Count Vectorizer parameter analyzer. The default is 'word'.
        sw : str, optional
            Count Vectorizer parameter stop_words. The default is 'english'.
        lc : BOOL, optional
            Count Vectorizer parameter lowercase. The default is True.
        bi : BOOL, optional
            Count Vectorizer parameter binary. The default is False.
        st: BOOL, optional
            Count Vectorizer parameter stemmer. The default is False.
        rare_word : int, optional
            Set the threshold to remove typo. The default is 10.

        Returns
        -------
        None.

        '''
        if st:
            Vec = CountVectorizer(analyzer = ana,
                                  stop_words = sw,
                                  lowercase = lc,
                                  binary = bi,
                                  tokenizer = TrainANN.stem)
        else:
            Vec = CountVectorizer(analyzer = ana,
                                  stop_words = sw,
                                  lowercase = lc,
                                  binary = bi)
            
        if full_query:
            temp_d = self.X_df['Title'].astype(str) + ' ' + self.X_df['Text_sub'].astype(str) + ' ' + self.X_df['Text_com'].astype(str)
        else:
            temp_d = self.X_df['Title'].astype(str) + ' ' + self.X_df['Text_sub'].astype(str)
        
        data_X_t = Vec.fit_transform(temp_d)
        
        features = Vec.get_feature_names()
        
        temp = pd.DataFrame(data_X_t.toarray(), columns = features)

        temp = temp.loc[:, temp.sum(axis = 0) > rare_word]
        
        self.X = temp
                    
    def embedding(self,max_words = 100,full_query = True,maxlen = 20):
        """
        an alternative method to
        :param max_words: max words to take from each reddit
        :param full_query: BOOL, optional
            Include comment or not. The default is True.
        :param maxlen: input length of the model
        :return:
        """

        ## to-do: add glove.6B file


        if full_query:
            temp_d = self.X_df['Title'].astype(str) + ' ' + self.X_df['Text_sub'].astype(str) + ' ' + self.X_df[
                'Text_com'].astype(str)
        else:
            temp_d = self.X_df['Title'].astype(str) + ' ' + self.X_df['Text_sub'].astype(str)


        tokenizer = Tokenizer(num_words=max_words)
        tokenizer.fit_on_texts(temp_d)
        sequences = tokenizer.texts_to_sequences(temp_d)

        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))
        self.X = pad_sequences(sequences, maxlen=maxlen)
        self.y = np.asarray(self.y[:,0])
        print('Shape of data tensor:', self.X.shape)
        print('Shape of label tensor:', self.y.shape)


        glove_dir = '/Users/irene/Downloads/glove.6B'
        embeddings_index = {}
        f = open(os.path.join(glove_dir, 'glove.6B.50d.txt'))
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        print('Found %s word vectors.' % len(embeddings_index))

        embedding_dim = 50
        self.embedding_dim = embedding_dim
        embedding_matrix = np.zeros((max_words, embedding_dim))
        for word, i in word_index.items():
            if i < max_words:
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector

        self.embedding_matrix = embedding_matrix




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
            
    def LSTM(self,plot = True, save_model= True,max_words = 100,maxlen = 20):
        '''
        Yunfei

        Returns
        -------
        None.

        '''
        model = Sequential()
        model.add(Embedding(max_words, self.embedding_dim, input_length=maxlen))

        # FIRST LAYER

        model.add(layers.LSTM(units = 32, dropout=0.2, recurrent_dropout=0.2))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.layers[0].set_weights([self.embedding_matrix])
        model.layers[0].trainable = False
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
        model.summary()

        training_samples = 150
        validation_samples = 20

        x_train = self.X[:training_samples]
        y_train = self.y[:training_samples]
        x_val = self.X[training_samples: training_samples + validation_samples]
        y_val = self.y[training_samples: training_samples + validation_samples]


        history = model.fit(x_train, y_train, epochs=10, batch_size=216, validation_data=(x_val, y_val))
        if plot:
            pass
            #report(history,title="LSTM")
        if save_model:
            model.save('LSTM.h5')


    
    def LSM(self):
        # Chevy
        pass
    
if __name__ == '__main__':
    df = TrainANN('sp500.txt', 'reddit_10.txt')

    df.embedding(max_words = 100,maxlen = 20)
    df.LSTM()