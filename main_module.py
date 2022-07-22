# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 15:03:42 2022

@author: isaac
"""

# import
import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as ss
import matplotlib.pyplot as plt

from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential, Input
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

class StepTwo():
    def utils(self, df):
        '''
        This function was meant to compact all the pandas inspect
        function

        Parameters
        ----------
        df : DataFrame
            Take DataFrame for inspection.

        Returns
        -------
        desc : DataFrame report
            pandas DataFrame describe.

        '''
        print('\n','- -'*5, '\nInfo\n', '- -'*5)
        df.info()
        desc = df.describe().T
        print('\n','- -'*5, '\nMissing Value\n', '- -'*5)
        print(df.isna().sum())
        print('\nDuplicated :',df.duplicated().sum())
        return desc
    
    
    def plot_categ(self, df):
        '''
        Countplot for categorical dataset

        Parameters
        ----------
        df : DataFrame
            Only accepts categorical DataFrame.

        Returns
        -------
        None.

        '''
        try:
            for i in df.columns:
              plt.figure()
              sns.countplot(x=df[i])
              plt.show()
        except:
            plt.figure()
            sns.countplot(x=df)
            plt.show()
    
    
    def plot_conti(self, df):
        '''
        Distplot for continous dataset

        Parameters
        ----------
        df : DataFrame
            Only accepts continous DataFrame.

        Returns
        -------
        None.

        '''
        try:
            for i in df.columns:
              plt.figure()
              sns.distplot(df[i])
              plt.show()
        except:
            plt.figure()
            sns.distplot(df)
            plt.show()
            
            
    def box_conti(self, df):
        '''
        Boxplot for detecting outliers

        Parameters
        ----------
        df : DataFrame
            Create a boxplot for inspection.

        Returns
        -------
        None.

        '''
        try:
            for i in df.columns:
              plt.figure()
              sns.boxplot(y=df[i])
              plt.show()
        except:
            plt.figure()
            sns.boxplot(y=df)
            plt.show()
            
    # inspect relation between two columns
    def groupby_plot(self, col1, col2, df):
        '''
        Create a groupby to inspect relation

        Parameters
        ----------
        col1 : str
            Desired columns name.
        col2 : str
            Desired columns name.
        df : DataFrame
            DataFrame to be inspect.

        Returns
        -------
        None.

        '''
        plt.figure()
        df.groupby([col1, col2]).agg({col1:'count'}).plot(kind='bar')
        plt.ylabel(col2)
        plt.show()


class StepThree():
    # label category
    def label_encoder(self, df, target):
        '''
        To encode object to numeric

        Parameters
        ----------
        df : DataFrame
            Can be filled with Nans.
        target : str
            To exclude target from encode.

        Returns
        -------
        df : DataFrame
            Object into numeric.

        '''
        le = LabelEncoder()
        
        for i in df.columns:
            if i == target:
                continue
            else:
                temp_ = df[i]
                temp_[temp_.notnull()] = le.fit_transform(temp_[temp_.notnull()])
                df[i] = pd.to_numeric(temp_, errors='coerce')
                name = i.upper()
                PICKLE_SAVE_PATH = os.path.join(os.getcwd(), 'models', name+'_ENCODER.pkl')
                with open(PICKLE_SAVE_PATH, 'wb') as file:
                    pickle.dump(le, file)
        return df
    

    def one_hot_encoder(self, y):
        '''
        Only for Deep Learning output

        Parameters
        ----------
        y : DataFrame
            Target DataFrame.

        Returns
        -------
        y : Array
            One_Hot_Encoded results.

        '''
        name = y.name.upper()
        ohe = OneHotEncoder(sparse=False)
        y = ohe.fit_transform(np.expand_dims(y, axis=-1))
        OHE_SAVE_PATH = os.path.join(os.getcwd(), 'models', name+'_ENCODER.pkl')
        with open(OHE_SAVE_PATH, 'wb') as file:
            pickle.dump(ohe, file)
        return y
    
    
    def imputer(self, df, col_name):
        '''
        To fill missing values

        Parameters
        ----------
        df : DataFrame
            DataFrame without str.
        col_name : list
            To rename the columns for DataFrame.

        Returns
        -------
        df : DataFrame
            Imputed DataFrame.

        '''
        knn_imputer = KNNImputer()
        df = knn_imputer.fit_transform(df)
        df = pd.DataFrame(df)
        df.columns = col_name
        return df
    
    
    def r_down(self, r_list, df):
        for i in r_list:
            df[i] = np.floor(df[i])
        return df


class StepFour():
    def cramers_corrected_stat(self, confusion_matrix):
        '''
        calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328

        Parameters
        ----------
        confusion_matrix : DataFrame
            Concated DataFrame.

        Returns
        -------
        int
            Correlation between each categorical variables.

        '''
        chi2 = ss.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum()
        phi2 = chi2/n
        r,k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))
    
    # categorical vs categorical
    def catvcat(self, categ, target, df, threshold, feature_score):
        '''
        To inspect relation between categorical

        Parameters
        ----------
        categ : list
            Categorical list.
        target : str
            Toward the target correlation.
        df : DataFrame
            Cleaned DataFrame.
        threshold : int
            Scores to select the best features.
        feature_score : list
            Extract the choosen features.

        Returns
        -------
        feature_score : list
            Extract the choosen features.

        '''
        for i in categ:
            cm_ = pd.crosstab(df[i], df[target]).to_numpy()
            print(i, StepFour().cramers_corrected_stat(cm_))
            if StepFour().cramers_corrected_stat(cm_) > threshold:
                feature_score.append(i)
        return feature_score
                

    # categorical vs continous
    def catvcon(self, conti, target, df, threshold, feature_score):
        '''
        Using Logistic Regression for feature selection

        Parameters
        ----------
        conti : list
            List of continous name.
        target : str
            To inspect relation between features.
        df : DataFrame
            Cleaned DataFrame.
        threshold : int
            Scores to select the best features.
        feature_score : list
            Extract the choosen features.

        Returns
        -------
        feature_score : list
            Extract the choosen features.

        '''
        for i in conti:
            lr=LogisticRegression()
            lr.fit(np.expand_dims(df[i],axis=-1),df[target])
            print(i, lr.score(np.expand_dims(df[i],axis=-1),df[target]))
            if lr.score(np.expand_dims(df[i],axis=-1),df[target]) > threshold:
                feature_score.append(i)
        return feature_score
                
                
class StepFive():
    def scaler(self, df, col_name):
        '''
        Scaler for input Deep Learning

        Parameters
        ----------
        df : DataFrame
            Features DataFrame.
        col_name : list
            List of features name.

        Returns
        -------
        df : DataFrame
            Scaled DataFrame.

        '''
        mms = MinMaxScaler()
        df = mms.fit_transform(df)
        df = pd.DataFrame(df, index=None)
        df.columns = col_name
        MMS_SAVE_PATH = os.path.join(os.getcwd(), 'models', 'MMS_ENCODER.pkl')
        with open(MMS_SAVE_PATH, 'wb') as file:
            pickle.dump(mms, file)
        return df
    
class SimpleModel():
    def model_number(self, input_shape, nb_class, hidden, cell):
        '''
        Deep Learning Sequential method

        Parameters
        ----------
        input_shape : array
            X_train input.
        nb_class : int
            Deeplearning output.
        hidden : TYPE
            DESCRIPTION.
        cell : TYPE
            DESCRIPTION.

        Returns
        -------
        model : Deep Learning
            Models.

        '''
        model = Sequential()
        model.add(Input(shape=input_shape))
          
        for i in range(hidden):
          model.add(Dense(cell, activation='relu'))
          model.add(BatchNormalization())
          model.add(Dropout(0.2))
          
        model.add(Dense(nb_class, activation='softmax'))
        return model

    
    # call backs
    def callbacks(self, LOGS_PATH, patience):
        '''
        Callbacks for TensorBoard

        Parameters
        ----------
        LOGS_PATH : Path
            Save Logs.
        patience : int
            Early callbacks.

        Returns
        -------
        list
            Input for model training.

        '''
        tensorboard_callbacks = TensorBoard(log_dir=LOGS_PATH, histogram_freq=1)
        early_callback = EarlyStopping(monitor='val_loss', patience=patience)
        return [early_callback, tensorboard_callbacks]

    
    def model_analysis(self, training, validation, hist):
        '''
        Plot graph for training and validation

        Parameters
        ----------
        training : str
            Input for plot.
        validation : str
            Input for plot.
        hist : array
            To plot the training and validation.

        Returns
        -------
        None.

        '''
        plt.figure()
        plt.plot(hist.history[training])
        plt.plot(hist.history[validation])
        plt.xlabel('epouch')
        plt.legend(['Training_'+training , 'Validation_'+training])
        plt.show()
        

    def model_utils(self, y_true, y_pred, labels):
        '''
        Utilities metrics evaluation

        Parameters
        ----------
        y_true : array
            y_test array.
        y_pred : array
            model prediction.
        labels : TYPE
            Output names.

        Returns
        -------
        None.

        '''
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_true, axis=1)
        cm = confusion_matrix(y_true, y_pred)
        cr = classification_report(y_true, y_pred)
        print(cr)
        
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap=plt.cm.Blues)
        plt.show













