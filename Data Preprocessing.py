#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.python.framework import ops
from tensorflow.keras import backend as K


# In[ ]:


def dataset_clean( dataset ):
    #  cleaning databases
    dataset = dataset.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
    dataset = dataset.drop_duplicates(subset=None, keep='first', inplace=False)

    return dataset


# In[ ]:


#Loading dataset
col_names=pd.read_csv('Field Names.csv')
kdd_train=pd.read_csv('KDDTrain+.csv')
kdd_test=pd.read_csv('KDDTest+.csv')

kdd_train=dataset_clean(kdd_train)
kdd_test=dataset_clean(kdd_test)

X_train=kdd_train.iloc[:, :41]
X_test=kdd_test.iloc[:, :41]
Y_train=kdd_train.iloc[:, 41:42]
Y_test=kdd_test.iloc[:, 41:42]

#Setting column names
X_train.columns=['duration', 'protocol_type', 'service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate']
X_test.columns=['duration', 'protocol_type', 'service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate']
Y_train


# In[ ]:


#Label Encoding
def label_encoding(data):
    le = LabelEncoder()
  
    data['protocol_type']= le.fit_transform(data['protocol_type'])
    data['service']= le.fit_transform(data['service'])
    data['flag']= le.fit_transform(data['flag'])

    return data


# In[ ]:


def ohe_normalize(data):
    
    # Define which columns should be encoded vs scaled
    columns_to_encode = data.iloc[:, 1:4]
    columns_to_scale1  = ['duration']
    columns_to_scale2  = data.iloc[:, 4:]
    
    #print(columns_to_encode)
    

    # Instantiate encoder/scaler
    scaler = StandardScaler()
    ohe    = OneHotEncoder(sparse=False)

    # Scale and Encode Separate Columns
    scaled_columns1  = scaler.fit_transform(data[columns_to_scale1]) 
    encoded_columns =  ohe.fit_transform(columns_to_encode)
    scaled_columns2  = scaler.fit_transform(columns_to_scale2)

    # Concatenate (Column-Bind) Processed Columns Back Together
    processed_data = np.concatenate([encoded_columns, scaled_columns2], axis=1)
    final=np.concatenate([scaled_columns1, processed_data], axis=1)
    
    return final


# In[ ]:


X_train=label_encoding(X_train)
X_train=ohe_normalize(X_train)


# In[ ]:


#transforming X_train dataset into a tensor
X_train=tf.convert_to_tensor(X_train)
X_train=tf.reshape(X_train, [125972,122,1,1])
X_train=tf.keras.layers.ZeroPadding2D(padding=(3,0))(X_train)


X_train=tf.reshape(X_train, [125972,8,8,2])
Y_train=tf.convert_to_tensor(Y_train.to_numpy())
Y_train=tf.reshape(Y_train, [125972,1])
Y_train


# In[ ]:


# Creating CNN Layers


def CNN_Model():
    model = tf.keras.Sequential([
        #tf.keras.layers.ZeroPadding2D(padding=3, input_shape=(64, 64, 3)),
    tf.keras.layers.Conv2D(20 , (7,7), padding='same',activation= 'tanh'), #Conv 1
    tf.keras.layers.BatchNormalization(axis=3), #Norm 1
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)), #MaxPool 1
         
    tf.keras.layers.Conv2D(40 , (7,7), padding='same',activation= 'tanh'), #Conv2
    tf.keras.layers.BatchNormalization(axis=3), #Norm 2
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)), #MaxPool 2
         
    tf.keras.layers.Conv2D(60 , (7,7), padding='same',activation= 'tanh'), #Conv 3
    tf.keras.layers.BatchNormalization(axis=3), #Norm 3
    #tf.keras.layers.Dense(1, activation='sigmoid'), #tanh 3
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)), #MaxPool 3
         
        
    #tf.keras.layers.ReLU(),
        
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid')
        
        
        
        
            
            # YOUR CODE ENDS HERE
    ])
    
    return model



#def recall_m(y_true, y_pred):
 #   true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  #  possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
   # recall = true_positives / (possible_positives + K.epsilon())
    #return recall

#def precision_m(y_true, y_pred):
 #   true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  #  predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
   # precision = true_positives / (predicted_positives + K.epsilon())
    #return precision

#def f1_m(y_true, y_pred):
 #   precision = precision_m(y_true, y_pred)
  #  recall = recall_m(y_true, y_pred)
   # return 2*((precision*recall)/(precision+recall+K.epsilon()))


# In[ ]:


model=CNN_Model()
model.compile(optimizer='adam',
loss='binary_crossentropy',
metrics=['accuracy'])


# In[ ]:


model.fit(X_train, Y_train, epochs=2, batch_size=10,verbose=1)


# In[ ]:


model.summary()


# In[ ]:




