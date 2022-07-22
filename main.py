# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 09:20:13 2022

@author: isaac
"""

import os
import datetime
import numpy as np
import pandas as pd

from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from main_module import StepTwo, StepThree, StepFour, StepFive, SimpleModel
#%% PATH

CSV_PATH = os.path.join(os.getcwd(), 'datasets', 'Train.csv')
LOGS_PATH = os.path.join(os.getcwd(), 'logs', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

#%% Step 1) Data Loading

# load dataframe
df = pd.read_csv(CSV_PATH)

#%% Step 2) Data Inspection & Visualization

step2 = StepTwo() # initialize the function

# compact utilities from main_module
desc = step2.utils(df) 

# check the shape
df.shape # row: 31647, columns:18

# identified which is categorical and continous
print('object :',(df.dtypes==object).sum())
print('not object :',(df.dtypes!=object).sum())

# inspect object type and non-object type. to determine whether categorical
# or non-categorical
df_obj = df.loc[:, (df.dtypes==object)]
df_num = df.loc[:, (df.dtypes!=object)]

# finalize the categorical and continous data
# categorical
categ = list(df_obj.drop(columns=['id']).columns)
categ.append(df['term_deposit_subscribed'].name)
categ.append(df['day_of_month'].name)

# continous
conti = list(df.drop(columns=categ).drop(columns=['id']).columns)

# if check == 1, all columns are properly distributed, because id was drop
check = len(df.columns) - len(categ) - len(conti)

# inspect categorical
df_categ = df[categ]
desc = step2.utils(df_categ) # ignore duplicated as we checked before 0 duplicated

# inspect continous
df_conti = df[conti]
desc = step2.utils(df_conti)

# categorical data plot
step2.plot_categ(df[categ])

# continous data plot
step2.plot_conti(df[conti])

# boxplot for detecting outliers
step2.box_conti(df[conti])

# negative number detected
df_neg = df.loc[(df.balance<0)]
step2.plot_categ(df_neg['term_deposit_subscribed'])


print('\n','- -'*5, '\nImbalanced\n', '- -'*5)
print(round(100*df['term_deposit_subscribed'].value_counts() / len(df), 2))

# objective : deep learning model to predict the outcome of the campaign
# the target is very unbalanced

# inspect the relation between term_deposit_subscribed and categorical
for i in categ:
    step2.groupby_plot('term_deposit_subscribed', i, df)

# print missing value percentage
print('\n','- -'*5, '\nMissing Value Percentage\n', '- -'*5)
print(round(100 * df.isna().sum() / len(df), 2))

# no duplicated
# large pool of missing data
# outliers

desc = step2.utils(df) # last recheck

#%% Step 3) Data Cleaning

# making a savepoint for references
df2 = df.copy() 

# remove unused columns
df2.drop(columns=['id', 'days_since_prev_campaign_contact'], inplace=True)

# update the continous columns name
conti.remove('days_since_prev_campaign_contact')

# the reason of dropping id : irrelevant while days_since_prev_campaign_contact
# possessed 80% NaNs

# recheck the shape after dropping
df2.shape 

# recheck using compact utils from main_module
desc = step2.utils(df2[conti])

# outliers detection
step2.box_conti(df2[conti])

# label encoding
step3 = StepThree()
df2[categ] = step3.label_encoder(df[categ], 'term_deposit_subscribed')

# recheck data
desc = step2.utils(df2) 
desc = step2.utils(df2[categ])

# one hot endcoding for target
y = step3.one_hot_encoder(df2['term_deposit_subscribed'])

# copy the columns name from latest dataframe
col_name = df2.columns

# saving inspection before impute
desc_before = step2.utils(df2)

# imputing missing value
df2 = step3.imputer(df2, col_name)

# based on the missing value before, down all relevent columns
r_list = ['customer_age', 'marital', 'personal_loan', 
          'last_contact_duration', 'num_contacts_in_campaign']

df2 = step3.r_down(r_list, df2) # function for rounding down

desc = step2.utils(df2)
desc_dif = desc_before - desc

#%% Step 4) Feature Selection
# create a checkpoint
df3 = df2.copy()
step4 = StepFour()

# emtpy list for collecting feature score to target
feature_score = []

# categorical and categorical selection
print('\n','- -'*5, '\n Categorical vs Categorical \n', '- -'*5)
feature_score = step4.catvcat(categ, 'term_deposit_subscribed', df3, 0.5, feature_score)

# categorical and continous selection
print('\n','- -'*5, '\n Categorical vs Continous \n', '- -'*5)
feature_score = step4.catvcon(conti, 'term_deposit_subscribed', df3, 0.5, feature_score)

# output the whole feature scores
print('\n','- -'*5, '\n Features Score \n', '- -'*5)
print(feature_score)

# separate the X and y for train split test
X = df3[feature_score].drop(columns=['term_deposit_subscribed'])
y = y

#%% Step 5) Data Preprocessing
step5 = StepFive()
col_name = X.columns # collect the remaining columns name

# scale X
X = step5.scaler(X, col_name)

# split train and test data
X_test, X_train, y_test, y_train = train_test_split(X, y, 
                                                    test_size=0.3, 
                                                    random_state=123)

#%% Model

c_model = SimpleModel() # initialize function

# create a input and output shape
input_shape = np.shape(X_train)[1:]
nb_class = len(np.unique(y_train, axis=0)) # alternative len(df.target.unique)

# create model, flexible to increase layers and network
model_1 = c_model.model_number(input_shape, nb_class, hidden=2, cell=64)
model_1.summary()
model_1.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['acc'])

# model plot
plot_model(model_1,
           show_layer_names=True,
           show_shapes=True)

# callbacks
early_callback, tensorboard_callbacks = c_model.callbacks(LOGS_PATH, 
                                                          patience=10)

# model training
hist = model_1.fit(X_train, y_train,
                   epochs=50,
                   batch_size=128,
                   validation_data=(X_test, y_test),
                   callbacks=[early_callback, tensorboard_callbacks])

#%% Model evaluation

print('\n','- -'*5, '\nTraining and Validation keys\n', '- -'*5)
print(hist.history.keys()) # taking the keys to plot

# plotting model training and validation
c_model.model_analysis('loss', 'val_loss', hist)
c_model.model_analysis('acc', 'val_acc', hist)


#%% Model evaluation

# directly take the unique = [0 or 1]
# labels = df3.term_deposit_subscribed.unique()
labels = ['No Subscribe', 'Subscribed']
y_pred = model_1.predict(X_test)

# check model metrics performances
print('\n','- -'*5, '\nMatrix Test Score keys\n', '- -'*5)
c_model.model_utils(y_test, y_pred, labels)

#%% model saving
MODEL_PATH = os.path.join(os.getcwd(), 'models', 'model.h5')
model_1.save(MODEL_PATH)

#%% Conclusion

# EDA
# 1. There is negative value in balance, outliers is present, many missing
# value and finally imbalanced between datasets

# Feature Selection
# 1. KNN imputer was used for filling the nans value, categorical converted
# into labels, while target converted via one hot encoding. Selection between
# feature were made into pipeline like concept where if else statement used
# to select which highly correlated with target

# Model
# 1. Created model using sequential method. Input and output highly depended
# on X_train and y_train. Layers were created using for loop, you may increase
# it and lastly network is also changeable.

# 2. Although, accuracy shows pleasing score, f1-score for seems to be less
# reliable.

# Testing phases
# 1. Tested with 2 layer, from network = 16 till 128, no significant results
# 2. Removing outliers and create a balanced data using SMOTE may increase
# the performances













