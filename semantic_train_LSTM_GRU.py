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
from keras.regularizers import l2



## MERGE CODE ##
# please paste the package, embedding(), LSTM(), GRU() and "if name == main" parts to the training script


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




    def embedding(self,max_words = 10000,full_query = True,maxlen = 250,glove_dir = '/Users/irene/Downloads/glove.6B'):
        """
        an alternative method to vectorization, pre-trained GloVe representation download needed 'https://github.com/stanfordnlp/GloVe'
        :param max_words: max words to take from each reddit
        :param full_query: BOOL, optional
            Include comment or not. The default is True.
        :param maxlen: input length of the model
        :param glove_dir: GloVe pretrained representation directory
        :return:
        """

        # Concat columns
        if full_query:
            temp_d = self.X_df['Title'].astype(str) + ' ' + self.X_df['Text_sub'].astype(str) + ' ' + self.X_df[
                'Text_com'].astype(str)
        else:
            temp_d = self.X_df['Title'].astype(str)


        # Tokenization
        tokenizer = Tokenizer(num_words=max_words)
        tokenizer.fit_on_texts(temp_d)
        sequences = tokenizer.texts_to_sequences(temp_d)

        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))
        self.X = pad_sequences(sequences, maxlen=maxlen)
        self.y = np.asarray(self.y[:,0])
        print('Shape of data tensor:', self.X.shape)
        print('Shape of label tensor:', self.y.shape)


        # Load in pre-trained model and save the embedding weights into a matrix
        embeddings_index = {}
        f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        print('Found %s word vectors.' % len(embeddings_index))

        embedding_dim = 100
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
            
    def LSTM(self,plot = True, save_model= True,max_words = 10000,maxlen = 250,shuffle= True):
        """
        LSTM model
        :param plot: whether to show the plots
        :param save_model: whether to save the model
        :param max_words: max words uses as features
        :param maxlen: input length of the embedding layer
        :param shuffle: whether to shuffle the index
        :return: (plots and saved models)
        """

        # Split X and y data
        indices = np.arange(self.X.shape[0])
        if shuffle == True:
            np.random.shuffle(indices)
            self.X = self.X[indices]
            self.y = self.y[indices]

        f_train = 0.6
        f_val   = 0.2
        CUT1 = int(f_train*self.X.shape[0]);
        CUT2 = int((f_train+f_val)*self.X.shape[0]);

        train_idx = indices[:CUT1]
        val_idx   = indices[CUT1:CUT2]
        test_idx  = indices[CUT2:]
        x_train, y_train = self.X[train_idx, :], self.y[train_idx]
        x_val, y_val = self.X[val_idx, :], self.y[val_idx]
        x_test, y_test = self.X[test_idx, :], self.y[test_idx]
        print('------PARTITION INFO---------')
        print("x_train shape:", x_train.shape)
        print("y_train shape:", y_train.shape)
        print("x_val shape:", x_val.shape)
        print("y_val shape:", y_val.shape)
        print("x_test shape:", x_test.shape)
        print("y_test shape:", y_test.shape)

        # build model
        model = Sequential()
        model.add(Embedding(max_words, self.embedding_dim, input_length=maxlen))
        model.add(layers.Conv1D(32, 7, activation='relu'))
        model.add(layers.MaxPooling1D(3))
        model.add(layers.Conv1D(32, 5, activation='relu'))
        model.add(layers.LSTM(units = 64, dropout=0.2, recurrent_dropout=0.2))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.layers[0].set_weights([self.embedding_matrix])
        model.layers[0].trainable = False
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
        model.summary()


        # train model
        history = model.fit(x_train, y_train, epochs=20, batch_size=200, validation_data=(x_val, y_val))

        # print and plot results
        train_bce = history.history['loss']
        val_bce   = history.history['val_loss']
        train_acc = history.history['acc']
        val_acc   = history.history['val_acc']

        min_train_bce = np.min(train_bce)
        min_val_bce = np.min(val_bce)
        min_train_bce_epoch = np.argmin(train_bce) + 1
        min_val_bce_epoch = np.argmin(val_bce) + 1
        max_train_acc = np.max(train_acc)
        max_val_acc = np.max(val_acc)
        max_train_acc_epoch = np.argmax(train_acc) + 1
        max_val_acc_epoch = np.argmax(val_acc) + 1

        print('-------Results-------')
        print('The minimum training binary cross-entropy is', round(min_train_bce, 2), 'and it occurs at epoch #',
              min_train_bce_epoch)
        print('The minimum validation binary cross-entropy is', round(min_val_bce, 2), 'and it occurs at epoch #',
              min_val_bce_epoch)
        print('The maximum training accuracy is {}% and it occurs at epoch # {}.'.format(round(max_train_acc*100, 2),
                                                                                         max_train_acc_epoch))
        print('The maximum validation accuracy is {}% and it occurs at epoch # {}.'.format(round(max_val_acc*100, 2),
                                                                                           max_val_acc_epoch), '\n')
        if plot:
            # TRAINING AND VALIDATION LOSS PLOT
            epochs = range(1, len(train_bce) + 1)
            plt.plot(epochs, train_bce, label = 'Training loss')
            plt.plot(epochs, val_bce, label = 'Validation loss')
            plt.title('Training & Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()

            # TRAINING AND VALIDATION ACCURACY PLOT
            epochs = range(1, len(train_acc) + 1)
            plt.plot(epochs, train_acc, label = 'Training accuracy')
            plt.plot(epochs, val_acc, label = 'Validation accuracy')
            plt.title('Training & Validation Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.show()




        if save_model:
            model.save('LSTM.h5')

    def GRU(self,plot = True, save_model= True,max_words = 10000,maxlen = 250,shuffle= False):
        """
        GRU model
        :param plot: whether to show the plots
        :param save_model: whether to save the model
        :param max_words: max words uses as features
        :param maxlen: input length of the embedding layer
        :param shuffle: whether to shuffle the dataset
        :return: (plots and model saved)
        """

        # split data i
        indices = np.arange(self.X.shape[0])
        if shuffle == True:

            np.random.shuffle(indices)
            self.X = self.X[indices]
            self.y = self.y[indices]

        f_train = 0.6
        f_val   = 0.4
        CUT1 = int(f_train*self.X.shape[0]);
        CUT2 = int((f_train+f_val)*self.X.shape[0]);


        train_idx = indices[:CUT1]
        val_idx   = indices[CUT1:CUT2]
        test_idx  = indices[CUT2:]
        x_train, y_train = self.X[train_idx, :], self.y[train_idx]
        x_val, y_val = self.X[val_idx, :], self.y[val_idx]
        print('------PARTITION INFO---------')
        print("x_train shape:", x_train.shape)
        print("y_train shape:", y_train.shape)
        print("x_val shape:", x_val.shape)
        print("y_val shape:", y_val.shape)



        # build model

        model = Sequential()
        model.add(layers.Embedding(max_words, self.embedding_dim, input_length=maxlen))
        model.add(layers.Conv1D(32, 7, activation='relu',bias_regularizer=l2(0.01)))
        model.add(layers.MaxPooling1D(3))
        model.add(layers.Conv1D(32, 5, activation='relu',bias_regularizer=l2(0.01)))
        model.add(layers.GRU(32, dropout=0.2, recurrent_dropout=0.5,bias_regularizer=l2(0.01)))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.layers[0].set_weights([self.embedding_matrix])
        model.layers[0].trainable = False
        model.summary()
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

        # train model
        history = model.fit(x_train, y_train, epochs=10, batch_size=200, validation_data=(x_val, y_val),
                            verbose=1)
        train_bce = history.history['loss']
        val_bce   = history.history['val_loss']
        train_acc = history.history['acc']
        val_acc   = history.history['val_acc']

        # show results
        min_train_bce = np.min(train_bce)
        min_val_bce = np.min(val_bce)
        min_train_bce_epoch = np.argmin(train_bce) + 1
        min_val_bce_epoch = np.argmin(val_bce) + 1
        max_train_acc = np.max(train_acc)
        max_val_acc = np.max(val_acc)
        max_train_acc_epoch = np.argmax(train_acc) + 1
        max_val_acc_epoch = np.argmax(val_acc) + 1


        print('-------Results-------')
        print('The minimum training binary cross-entropy is', round(min_train_bce, 2), 'and it occurs at epoch #',
              min_train_bce_epoch)
        print('The minimum validation binary cross-entropy is', round(min_val_bce, 2), 'and it occurs at epoch #',
              min_val_bce_epoch)
        print('The maximum training accuracy is {}% and it occurs at epoch # {}.'.format(round(max_train_acc*100, 2),
                                                                                         max_train_acc_epoch))
        print('The maximum validation accuracy is {}% and it occurs at epoch # {}.'.format(round(max_val_acc*100, 2),
                                                                                           max_val_acc_epoch), '\n')

        if plot:
            # TRAINING AND VALIDATION LOSS PLOT
            epochs = range(1, len(train_bce) + 1)
            plt.plot(epochs, train_bce, label = 'Training loss')
            plt.plot(epochs, val_bce, label = 'Validation loss')
            plt.title('Training & Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()

            # TRAINING AND VALIDATION ACCURACY PLOT
            epochs = range(1, len(train_acc) + 1)
            plt.plot(epochs, train_acc, label = 'Training accuracy')
            plt.plot(epochs, val_acc, label = 'Validation accuracy')
            plt.title('Training & Validation Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.show()

        # save model
        if save_model:
            model.save('GRU.h5')


    
    def LSM(self):
        # Chevy
        pass
    
if __name__ == '__main__':
    df = TrainANN('sp500.txt', 'reddit_10.txt')
    df.embedding(max_words = 100000,full_query = False,maxlen = 500,glove_dir = '/Users/irene/Downloads/glove.6B')
    df.LSTM(max_words = 10000,maxlen = 250,shuffle=False)
    df.GRU(max_words = 100000,maxlen = 500,shuffle = False)


