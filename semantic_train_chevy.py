'''
Chevy Robertson (crr78@georgetown.edu)
ANLY 590: Neural Networks & Deep Learning
Group Semester Project: DFF Model Training & Results
11/30/2021
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
import pickle
import re
import tensorflow as tf
from datetime import datetime
from nltk.stem import PorterStemmer
from keras import preprocessing
from keras import models
from keras.models import Sequential
from keras import layers
from keras.callbacks import CSVLogger
from keras.datasets import imdb
from tensorflow.keras.optimizers import RMSprop
from sklearn.preprocessing import OneHotEncoder as OHE
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from statistics import mode 
from time import sleep
from text_data_processor import TextDataProcessor
from finance_data_collector import YahooData

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
                  rare_word = 10000):
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

        stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
             'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him',
             'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 
             'itself', 'they', 'them', 'their', 'theirs', 'themselves',
             'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
             'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
             'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
             'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
             'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
             'between', 'into', 'through', 'during', 'before', 'after',
             'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on',
             'off', 'over', 'under', 'again', 'further', 'then', 'once',
             'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
             'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
             'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
             'very', 'can', 'will', 'just', 'don', 'should', 'now', 'b', 'c',
             'd', 'e', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
             'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

        corpus = [mode(titles) for titles in self.X_df['Title']]

        '''
        bodies = []
        labels = []

        for date in range(len(self.X_df['Title'])):
            for title in range(len(self.X_df['Title'][date])):
                if self.X_df['Title'][date][title] == corpus[date]:
                    bodies.append(self.X_df['Text_com'][date][title])
                    labels.append(self.y[date])

        corpus = bodies
        '''

        semi_clean_chunks = []

        for chunk in corpus:
            try:
                words = chunk.split()
            except AttributeError:
                words = str(chunk).split()
            rm_nab = [re.compile('[^a-zA-Z]').sub('', word) for word in words]
            words_to_keep = []
            for word in rm_nab:
                word = word.lower()
                if not (word in stopwords):
                    words_to_keep.append(word)
            semi_clean_chunks.append(' '.join(words_to_keep))

        rm_ws = [chunk.lstrip().rstrip() for chunk in semi_clean_chunks]

        clean_chunks = []

        for chunk in rm_ws:
            clean_chunks.append(chunk)

        self.corpus = np.asarray(clean_chunks)
        # self.y = np.asarray(labels)


        '''
        data_X_t = Vec.fit_transform(temp_d)
        
        features = Vec.get_feature_names()
        
        temp = pd.DataFrame(data_X_t.toarray(), columns = features)

        temp = temp.loc[:, temp.sum(axis = 0) > rare_word]
        
        self.X = temp
        '''


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
            
    def LSTM(self):
        '''
        Yunfei

        Returns
        -------
        None.

        '''
        pass
    
    def DFF(self):

        '''
        negative_indices = []
        positive_indices = []
        count = 0
        for i in self.y:
            if i == 0.:
                negative_indices.append(count)
            else:
                positive_indices.append(count)
            count += 1

        corpus, labels = list(self.corpus), self.y
        '''

        rand_indices = np.random.RandomState(seed=0).permutation(self.corpus.shape[0])
        rand_idx = rand_indices[:self.corpus.shape[0]]
        corpus, labels = list(self.corpus[rand_idx]), self.y[rand_idx]

        def form_dictionary(samples):
            token_index = {}
            for sample in samples:
                for word in sample.split():
                    if word not in token_index:
                        token_index[word] = len(token_index) + 1
            transformed_text=[]
            for sample in samples:
                tmp=[]
                for word in sample.split():
                    tmp.append(token_index[word])
                transformed_text.append(tmp)
            print("CONVERTED TEXT:", transformed_text)
            print("VOCABULARY-2 (SKLEARN): ", token_index)
            return [token_index, transformed_text]
            
        [vocab, x] = form_dictionary(corpus)

        def one_hot_encode(samples):
            max_length = 100
            results = np.zeros(shape=(len(samples), max_length, max(vocab.values())+1))
            for i, sample in enumerate(samples):
                for j, word in list(enumerate(sample.split()))[:max_length]:
                    index = vocab.get(word)
                    results[i, j, index] = 1.
            results=results[:,:,1:]
            return results

        def vectorize_sequences(sequences, dimension):
            results = np.zeros((len(sequences), dimension))
            for i, sequence in enumerate(sequences):
                results[i, sequence] = 1.
            return results

        x_ohe = one_hot_encode(corpus)
        # x_ohe = vectorize_sequences(x, len(vocab)+1)

        #----------------
        # HYPERPARAMETERS
        #----------------

        nodes_in = 16                                # number of input layer nodes
        nodes_hidden = 16                            # number of hidden layer nodes
        nodes_out = 1                                # number of output layer nodes
        ks = 3                                       # kernel size
        ps = 2                                       # pool size 
        act_in = 'linear'                            # activation function for input layer
        act_hidden = 'linear'                        # activation function for hidden layer
        act_out = 'sigmoid'                          # activation function for output layer
        shape_in = x_ohe.shape[1:]                   # shape of input for input layer
        ki = 'ones'                                  # specify kernel initializer
        bi = 'zeros'                                 # specify bias initializer
        opt = 'rmsprop'                              # optimizer to use
        loss_func = 'binary_crossentropy'            # loss function to use
        metrics = ['accuracy', keras.metrics.AUC()]  # use accuracy, AUC for model metrics
        num_epochs = 100                              # number of epochs
        b_size = 32                                  # batch size

        def build_cnn():
            model = models.Sequential()
            model.add(layers.Conv1D(nodes_in, ks, activation=act_in, input_shape=shape_in, kernel_initializer=ki,
                                    bias_initializer=bi))
            # model.add(layers.Dropout(0.2))
            model.add(layers.Flatten())
            model.add(layers.Dense(nodes_hidden, activation=act_hidden))
            model.add(layers.Dense(nodes_out, activation=act_out))
            model.compile(optimizer=opt, loss=loss_func, metrics=metrics)
            print(model.summary())
            return model
        
        '''
        def build_rnn():
            model = models.Sequential()
            model.add(layers.Embedding(len(vocab.keys()), nodes_in))
            model.add(layers.SimpleRNN(nodes_in))
            # model.add(layers.Dropout(0.5))
            model.add(layers.Dense(nodes_out, activation=act_out))
            model.compile(optimizer=opt, loss=loss_func, metrics=metrics)
            print(model.summary())
            return model
        '''

        model = build_cnn()
        # model = build_rnn()

        '''
        f_train = 0.6
        f_val   = 0.2
        negative_cut1 = int(f_train*len(negative_indices))
        negative_cut2 = int((f_train+f_val)*len(negative_indices))
        positive_cut1 = int(f_train*len(positive_indices))
        positive_cut2 = int((f_train+f_val)*len(positive_indices))
        train_idx = negative_indices[:negative_cut1] + positive_indices[:positive_cut1]
        val_idx = negative_indices[negative_cut1:negative_cut2] + positive_indices[positive_cut1:positive_cut2]
        test_idx = negative_indices[negative_cut2:] + positive_indices[positive_cut2:]
        x_train, y_train = x_ohe[train_idx, :], labels[train_idx]
        x_val, y_val = x_ohe[val_idx, :], labels[val_idx]
        x_test, y_test = x_ohe[test_idx, :], labels[test_idx]
        print('------PARTITION INFO---------')
        print("x_train shape:", x_train.shape)
        print("y_train shape:", y_train.shape)
        print("x_val shape:", x_val.shape)
        print("y_val shape:", y_val.shape)
        print("x_test shape:", x_test.shape)
        print("y_test shape:", y_test.shape)
        '''

        f_train = 0.6
        f_val   = 0.2
        rand_indices = np.random.RandomState(seed=0).permutation(x_ohe.shape[0])
        CUT1 = int(f_train*x_ohe.shape[0]); 
        CUT2 = int((f_train+f_val)*x_ohe.shape[0]);
        train_idx = rand_indices[:CUT1]
        val_idx   = rand_indices[CUT1:CUT2]
        test_idx  = rand_indices[CUT2:]
        x_train, y_train = x_ohe[train_idx, :], labels[train_idx]
        x_val, y_val = x_ohe[val_idx, :], labels[val_idx]
        x_test, y_test = x_ohe[test_idx, :], labels[test_idx]
        print('------PARTITION INFO---------')
        print("x_train shape:", x_train.shape)
        print("y_train shape:", y_train.shape)
        print("x_val shape:", x_val.shape)
        print("y_val shape:", y_val.shape)
        print("x_test shape:", x_test.shape)
        print("y_test shape:", y_test.shape)
        
        history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=b_size, validation_data=(x_val, y_val))

        train_bce = history.history['loss']          
        val_bce   = history.history['val_loss']      
        train_acc = history.history['accuracy']     
        val_acc   = history.history['val_accuracy'] 
        train_auc = history.history['auc']          
        val_auc   = history.history['val_auc']       

        min_train_bce = np.min(train_bce)
        min_val_bce = np.min(val_bce)
        min_train_bce_epoch = np.argmin(train_bce) + 1
        min_val_bce_epoch = np.argmin(val_bce) + 1
        max_train_acc = np.max(train_acc)
        max_val_acc = np.max(val_acc)
        max_train_acc_epoch = np.argmax(train_acc) + 1
        max_val_acc_epoch = np.argmax(val_acc) + 1
        max_train_auc = np.max(train_auc)
        max_val_auc = np.max(val_auc)
        max_train_auc_epoch = np.argmax(train_auc) + 1
        max_val_auc_epoch = np.argmax(val_auc) + 1

        print('-------Results-------')
        print('The minimum training binary cross-entropy is', round(min_train_bce, 2), 'and it occurs at epoch #',
              min_train_bce_epoch)
        print('The minimum validation binary cross-entropy is', round(min_val_bce, 2), 'and it occurs at epoch #',
              min_val_bce_epoch)
        print('The maximum training accuracy is {}% and it occurs at epoch # {}.'.format(round(max_train_acc*100, 2),
                                                                                         max_train_acc_epoch))
        print('The maximum validation accuracy is {}% and it occurs at epoch # {}.'.format(round(max_val_acc*100, 2),
                                                                                           max_val_acc_epoch), '\n')
        print('The maximum training AUC is {} and it occurs at epoch # {}.'.format(round(max_train_auc, 4),
                                                                                   max_train_auc_epoch))
        print('The maximum validation AUC is {} and it occurs at epoch # {}.'.format(round(max_val_auc, 4),
                                                                                     max_val_auc_epoch), '\n')

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

        '''
        #---------------------------------------
        # PARTITIONING, SHUFFLING, & NORMALIZING
        #---------------------------------------

        # calculate mean and standard deviation 
        x_mean = np.mean(self.X, axis = 0)
        x_std = np.std(self.X, axis = 0)

        # split data into training and testing
        f_train = 0.8
        rand_indices = np.random.RandomState(seed=0).permutation(self.X.shape[0])
        CUT_test = int(f_train*self.X.shape[0])
        train_idx = rand_indices[:CUT_test]
        test_idx = rand_indices[CUT_test:]
        x_train, y_train = df.X[train_idx], df.y[train_idx]
        x_test, y_test = df.X[test_idx], df.y[test_idx]

        # normalize
        x_train = (x_train - x_mean)/x_std
        x_test = (x_test - x_mean)/x_std

        # split training data into train & validation
        f_train = 0.75
        CUT_val = int(f_train*x_train.shape[0]) 
        train_idx = rand_indices[:CUT_val]
        val_idx   = rand_indices[CUT_val:CUT_test]
        x_train, y_train = df.X[train_idx], df.y[train_idx]
        x_val, y_val = df.X[val_idx], df.y[val_idx]
        print('------PARTITION INFO---------')
        print("x_train shape:", x_train.shape)
        print("y_train shape:", y_train.shape)
        print("x_val shape:", x_val.shape)
        print("y_val shape:", y_val.shape)
        print("x_test shape:", x_test.shape)
        print("y_test shape:", y_test.shape)

        #-------------------------
        # TRAINING DFF (MLP) MODEL
        #-------------------------

        # HYPERPARAMETERS
        nodes = 16
        act_in = 'sigmoid'
        act_hidden = 'relu'
        act_out = 'sigmoid'
        opt = 'rmsprop'
        loss_func = 'binary_crossentropy'
        metric = 'accuracy'
        num_epochs = 20
        size = 32
        # kr = 'l2'
        
        # specify input shape
        input_shape = (x_train.shape[1],)
        
        # BUILD & COMPILE MODEL
        def build_model():
            model = models.Sequential()
            model.add(layers.Dense(units = nodes, activation = act_in, input_shape = input_shape))
            model.add(layers.Dense(units = nodes, activation = act_hidden))
            model.add(layers.Dropout(0.2))
            model.add(layers.Dense(1, activation = act_out))
            model.compile(optimizer = opt, loss = loss_func, metrics = [metric])
            return model
        
        # instantiate a compiled model
        model = build_model()
        
        print('Training model...', '\n')
        
        # fit the model onto the training set, validate the model with the validation set
        history = model.fit(x_train, y_train, epochs = num_epochs, batch_size = size, validation_data = (x_val, y_val))
        
        # record binary cross-entropy and accuracy on the train and val sets
        train_bce = history.history['loss']
        val_bce = history.history['val_loss']
        train_acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        
        # record min binary cross-entropy and max accuracy for train and val, and epochs they occur at
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
        '''

        # before training final model, combine training and validation data so that we only predict on unseen test data
        
        # combine all input values from train and val
        x = np.vstack([x_train, x_val])
        
        # combine all output values from train and val
        y = np.vstack([y_train.reshape(y_train.shape[0], 1), y_val.reshape(y_val.shape[0], 1)])
        
        # train the final model and evaluate its performance on the test set
        model = build_cnn()
        model.fit(x, y, epochs = min_val_bce_epoch, batch_size = b_size)
        bce_test, acc_test, auc_test = model.evaluate(x_test, y_test)
        print('The binary cross-entropy of the model on the testing set is', round(bce_test, 2))
        print('The accuracy of the model on the testing set is {}%.'.format(round(acc_test*100, 2)))
        print('The AUC of the model on the testing set is {}.'.format(round(auc_test, 4)))

        # model.save('cnn_mod.h5')
        
        # make predictions from the train, val, and test data using the model
        yp_train = model.predict(x_train)
        yp_val   = model.predict(x_val)
        yp_test  = model.predict(x_test)
        
        # round the predicted probabilities to the nearest digit to indicate either 0 (neg pred) or 1 (pos pred)
        yp_train = np.around(yp_train)
        yp_val   = np.around(yp_val)
        yp_test  = np.around(yp_test)
        
        
        #-------- 
        # RESULTS
        #--------
                
        # TRAINING PREDICTIONS 
        train_TP = (confusion_matrix(y_train, yp_train)/y_train.shape[0])[1][1]*100
        train_TN = (confusion_matrix(y_train, yp_train)/y_train.shape[0])[0][0]*100
        train_FP = (confusion_matrix(y_train, yp_train)/y_train.shape[0])[0][1]*100
        train_FN = (confusion_matrix(y_train, yp_train)/y_train.shape[0])[1][0]*100
        
        # form a list of the categories for possible prediction types
        cats = ['True Positives', 'True Negatives', 'False Positives', 'False Negatives']
        
        # store the percentage of each prediction type
        pcts = [train_TP, train_TN, train_FP, train_FN]
        
        # PLOT TRAINING PREDICTIONS
        plt.bar(cats, pcts)
        plt.title('Model Results (Training Set)')
        plt.xlabel('Prediction Type')
        plt.ylabel('Percentage')
        plt.show()
        
        # VALIDATION PREDICTIONS 
        val_TP = (confusion_matrix(y_val, yp_val)/y_val.shape[0])[1][1]*100
        val_TN = (confusion_matrix(y_val, yp_val)/y_val.shape[0])[0][0]*100
        val_FP = (confusion_matrix(y_val, yp_val)/y_val.shape[0])[0][1]*100
        val_FN = (confusion_matrix(y_val, yp_val)/y_val.shape[0])[1][0]*100
        
        # form a list of the categories for possible prediction types
        cats = ['True Positives', 'True Negatives', 'False Positives', 'False Negatives']
        
        # store the percentage of each prediction type
        pcts = [val_TP, val_TN, val_FP, val_FN]
        
        # PLOT VALIDATION PREDICTIONS
        plt.bar(cats, pcts)
        plt.title('Model Results (Validation Set)')
        plt.xlabel('Prediction Type')
        plt.ylabel('Percentage')
        plt.show()        

        '''
        # TESTING PREDICTIONS

        # record the number of true and false positives in the testing set
        test_TP = (confusion_matrix(y_test, yp_test))[1][1]
        test_FP = (confusion_matrix(y_test, yp_test))[0][1]

        # use the sum of positive predictions to determine the proportion of true positives
        prop_TP = test_TP/(test_TP + test_FP)
        '''
 
        # TESTING PREDICTIONS
        test_TP = (confusion_matrix(y_test, yp_test)/y_test.shape[0])[1][1]*100
        test_TN = (confusion_matrix(y_test, yp_test)/y_test.shape[0])[0][0]*100
        test_FP = (confusion_matrix(y_test, yp_test)/y_test.shape[0])[0][1]*100
        test_FN = (confusion_matrix(y_test, yp_test)/y_test.shape[0])[1][0]*100
        
        # form a list of the categories for possible prediction types
        cats = ['True Positives', 'True Negatives', 'False Positives', 'False Negatives']
        
        # store the percentage of each prediction type
        pcts = [test_TP, test_TN, test_FP, test_FN]
        
        # PLOT TESTING PREDICTIONS
        plt.bar(cats, pcts)
        plt.title('Model Results (Testing Set)')
        plt.xlabel('Prediction Type')
        plt.ylabel('Percentage')
        plt.show()

        
if __name__ == '__main__':
    # df = pickle.load(open('tesla.txt', 'rb'))
    df = TrainANN('sp500.txt', 'reddit_10_small.txt')
    new_y = [i[1] for i in df.y]
    new_y = np.asarray(new_y).astype('float32')
    new_y = new_y.reshape(len(new_y), 1)
    df.y = new_y
    df.vectorize()
    # new_X = np.array([list(df.X.loc[i, :]) for i in range(len(df.X))])
    # df.X = new_X
    df.DFF()


