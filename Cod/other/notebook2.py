import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import decimate
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Conv1D, MaxPool1D, GlobalAvgPool1D, Dropout, BatchNormalization, Dense
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras.utils import np_utils
from keras.regularizers import l2


INPUT_LIB = 'heartbeat-sounds/'
SAMPLE_RATE = 44100
CLASSES = ['artifact', 'normal', 'murmur']
CODE_BOOK = {x:i for i,x in enumerate(CLASSES)}   
NB_CLASSES = len(CLASSES)


# ## Load the data
# We will need to preprocess the unclassified data, as they are marked NaN and also have incorrect file names. For now, we will also make all time series have equal length. We will do all this by defining element wise functions, that we can pass to Pandas.

# In[4]:


def clean_filename(fname, string):   
    file_name = fname.split('/')[1]
    if file_name[:2] == '__':        
        file_name = string + file_name
    return file_name


# In[5]:


def load_wav_file(name, path):
    _, b = wavfile.read(path + name)
    assert _ == SAMPLE_RATE
    return b


# In[6]:


def repeat_to_length(arr, length):
    """Repeats the numpy 1D array to given length, and makes datatype float"""
    result = np.empty((length, ), dtype = 'float32')
    l = len(arr)
    pos = 0
    while pos + l <= length:
        result[pos:pos+l] = arr
        pos += l
    if pos < length:
        result[pos:length] = arr[:length-pos]
    return result


# In[13]:


df = pd.read_csv(INPUT_LIB + 'set_a.csv')
df['fname'] = df['fname'].apply(clean_filename, string='Aunlabelledtest')
df['label'].fillna('unclassified')
df['time_series'] = df['fname'].apply(load_wav_file, path=INPUT_LIB + 'set_a/')    
df['len_series'] = df['time_series'].apply(len)
MAX_LEN = max(df['len_series'])
df['time_series'] = df['time_series'].apply(repeat_to_length, length=MAX_LEN) 


# In[15]:


df.head()


# ## Convert data to numpy arrays
# Let's leave the unclassified files for validation, and make a training set of the others. We will zero-pad the time series at the end to make the all the same length, then collect all in 2D numpy arrays, that can be used for neural network training.

# In[16]:


x_data = np.stack(df['time_series'].values, axis=0)


# In[39]:


x_data[0:5]


# Now, as explained in another [notebook][1], I will not use the classification from new_info['target'] but instead my own labels, with three classes 0=artifact, 1=normal/extrahls, and 2=murmur.
# 
# 
#   [1]: https://www.kaggle.com/toregil/d/kinguistics/heartbeat-sounds/misclassified-files-in-set-a/editnb "notebook"

# In[17]:


new_labels =[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2, 1, 1, 2, 1, 
             2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
             2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 
             1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 2, 1, 0, 2, 2, 1, 1, 1, 1, 1, 
             0, 1, 0, 1, 1, 1, 2, 1, 0, 1, 1, 1, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 
             1, 0, 0, 0, 0, 0, 0, 1, 0, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
new_labels = np.array(new_labels, dtype='int')
y_data = np_utils.to_categorical(new_labels)


# In[65]:


x_train, x_test, y_train, y_test, train_filenames, test_filenames =     train_test_split(x_data, y_data, df['fname'].values, test_size=0.25)


# We now downsample the data with what is in effect a very aggressive low pass filter. This is not needed for computational time, but it seems to improve generalization on this dataset. With more data, we should remove or reduce this step and instead add 2-5 extra convolution layers. The reason this works is probably that what you hear in the stethoscope is almost exclusively low frequency sounds, especially murmurs.

# In[66]:


x_train = decimate(x_train, 8, axis=1, zero_phase=True)
x_train = decimate(x_train, 8, axis=1, zero_phase=True)
x_train = decimate(x_train, 4, axis=1, zero_phase=True)
x_test = decimate(x_test, 8, axis=1, zero_phase=True)
x_test = decimate(x_test, 8, axis=1, zero_phase=True)
x_test = decimate(x_test, 4, axis=1, zero_phase=True)


# In[67]:


#Scale each observation to unit variance, it should already have mean close to zero.
x_train = x_train / np.std(x_train, axis=1).reshape(-1,1)
x_test = x_test / np.std(x_test, axis=1).reshape(-1,1)


# In[70]:


y_train[0]


# Keras wants the data two have a channel dimension, analogous to RGB for image recognition. Let's add this.

# In[64]:


x_train = x_train[:,:,np.newaxis]
x_test = x_test[:,:,np.newaxis]


# In[58]:


y_train[0]


# ##Train the model

# In[46]:


model = Sequential()
model.add(Conv1D(filters=4, kernel_size=9, activation='relu',
                input_shape = x_train.shape[1:],
                kernel_regularizer = l2(0.025)))
model.add(MaxPool1D(strides=4))
model.add(BatchNormalization())
model.add(Conv1D(filters=4, kernel_size=9, activation='relu',
                kernel_regularizer = l2(0.05)))
model.add(MaxPool1D(strides=4))
model.add(BatchNormalization())
model.add(Conv1D(filters=8, kernel_size=9, activation='relu',
                 kernel_regularizer = l2(0.1)))
model.add(MaxPool1D(strides=4))
model.add(BatchNormalization())
model.add(Conv1D(filters=16, kernel_size=9, activation='relu'))
model.add(MaxPool1D(strides=4))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Conv1D(filters=64, kernel_size=4, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Conv1D(filters=32, kernel_size=1, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.75))
model.add(GlobalAvgPool1D())
model.add(Dense(3, activation='softmax'))


# This version of the net has 5.000 parameters, and it could easily overfit our dataset of 125 time series if we let it run too long.

# In[33]:


def batch_generator(x_train, y_train, batch_size):
    """
    Rotates the time series randomly in time
    """
    x_batch = np.empty((batch_size, x_train.shape[1], x_train.shape[2]), dtype='float32')
    y_batch = np.empty((batch_size, y_train.shape[1]), dtype='float32')
    full_idx = range(x_train.shape[0])
    
    while True:
        batch_idx = np.random.choice(full_idx, batch_size)
        x_batch = x_train[batch_idx]
        y_batch = y_train[batch_idx]
    
        for i in range(batch_size):
            sz = np.random.randint(x_batch.shape[1])
            x_batch[i] = np.roll(x_batch[i], sz, axis = 0)
     
        yield x_batch, y_batch


# In[34]:


weight_saver = ModelCheckpoint('set_a_weights.h5', monitor='val_loss', 
                               save_best_only=True, save_weights_only=True)


# In[35]:


model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.8**x)


# In[36]:


hist = model.fit_generator(batch_generator(x_train, y_train, 8),
                   epochs=30, steps_per_epoch=1000,
                   validation_data=(x_test, y_test),
                   callbacks=[weight_saver, annealer],
                   verbose=2)


# Let's bring back the best weights, in case we overfitted.

# In[28]:


model.load_weights('set_a_weights.h5')


# ## Evaluation

# In[29]:


plt.plot(hist.history['loss'], color='b')
plt.plot(hist.history['val_loss'], color='r')
plt.show()
plt.plot(hist.history['acc'], color='b')
plt.plot(hist.history['val_acc'], color='r')
plt.show()


# In[30]:


y_hat = model.predict(x_test)
np.set_printoptions(precision=2, suppress=True)
for i in range(3):
    plt.plot(y_hat[:,i], c='r')
    plt.plot(y_test[:,i], c='b')
    plt.show()
    print(CLASSES[i])


# In[31]:


y_pred = np.argmax(y_hat, axis=1)
y_true = np.argmax(y_test, axis=1)
for i in range(len(y_true)):
    if y_pred[i] != y_true[i]:
        print("File: {}, Pred: {}, True: {}".format(
            test_filenames[i],
            CLASSES[y_pred[i]], CLASSES[y_true[i]]))
        plt.plot(x_test[i])
        plt.show()


# Not too bad. It found the artifacts and the mistakes were non-obvious ones.   If this dataset really comes from an iPhone app, it is truly impressive.
# 
#  I'd be really grateful if someone could double check my labelling, both on training and test set. Check the "New labels for set A" [notebook][1].
# 
# 
#   [1]: https://www.kaggle.com/toregil/d/kinguistics/heartbeat-sounds/new-labels-for-set-a

# I'll be back with an analysis of set B.
