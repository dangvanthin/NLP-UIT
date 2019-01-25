# -*- coding: utf-8 -*-
import pickle
import pandas as pd, re,string,os,io
from sklearn.feature_extraction import stop_words
from nltk.tokenize import word_tokenize
from keras.layers.convolutional import SeparableConv1D
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import nltk, csv, sys
from nltk.tokenize import word_tokenize
from keras.utils.vis_utils import plot_model
from sklearn.decomposition import PCA
import random as rn
from keras.constraints import max_norm
import tensorflow as tf
import keras
from keras.models import Model
from keras import optimizers
from keras.regularizers import l2
from keras.callbacks import *
from keras.engine import Layer
from keras import backend as K
from keras_utils import Capsule, AttentionWithContext,Attention
from keras.layers import *
from keras.optimizers import *
from gensim.models import KeyedVectors
import multiprocessing
from Layers import AttentionWeightedAverage
from sklearn.utils.class_weight import compute_class_weight


from data_process import *
os.environ['PYTHONHASHSEED'] = '0'                    
np.random.seed(1337)
rn.seed(1337)
tf.set_random_seed(1337)
max_cores = multiprocessing.cpu_count()
num_cores = max_cores - 2
num_CPU = 1
num_GPU = 1
session_conf = tf.ConfigProto(intra_op_parallelism_threads=num_cores, inter_op_parallelism_threads=num_cores,allow_soft_placement=True,device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


# read data after preprocessing steps
PIK = "document_X_train.dat"
with open(PIK, "rb") as f:
    document_X_train = pickle.load(f)

PIK2 = "document_Y_train.dat"
with open(PIK2, "rb") as f:
    document_Y_train = pickle.load(f)

PIK3 = "document_X_train_raw.dat"
with open(PIK3, "rb") as f:
    document_X_train_raw = pickle.load(f)

# get sequence length of input
xLengths = [len(word_tokenize(x)) for x in document_X_train]
h = sorted(xLengths)  #sorted lengths
maxlen_word = h[-1]
print("max input length is: ",maxlen_word)

# creat vocabulary
max_vocab_size = 10000
input_tokenizer = Tokenizer(max_vocab_size)
input_tokenizer.fit_on_texts(document_X_train)
input_vocab_size = len(input_tokenizer.word_index) + 1
word_index = input_tokenizer.word_index
print("input_vocab_size:",input_vocab_size)
totalX = np.array(pad_sequences(input_tokenizer.texts_to_sequences(document_X_train), maxlen=maxlen_word))


# creat word embedding for vocabulary
def generate_embedding(word_index, model_embedding,EMBEDDING_DIM):
  count = 0
  countNot = 0
  embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
  for word, i in word_index.items():
      try:
          embedding_vector = model_embedding[word]
      except:
          countNot +=1
          continue
      if embedding_vector is not None:
          count +=1
          embedding_matrix[i] = embedding_vector
  
  print('Number of words in pre-train embedding: ' + str(count))
  print('Number of words not in pre-train embedding: ' + str(countNot))
  return embedding_matrix


def focal_loss(y_true, y_pred, alpha, gamma=0.5):
    alpha = K.variable(alpha)
    pt = K.abs(1. - y_true - y_pred)
    pt = K.clip(pt, K.epsilon(), 1. - K.epsilon())
    return K.mean(-alpha * K.pow(1. - pt, gamma) * K.log(pt), axis=-1)
    

# read pretrain word embedding FastText
def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.fromiter(tokens[1:], dtype=np.float)
    return data

word_embedding_fastText = load_vectors('/home/thindv/ShareTask/Embedding/crawl-300d-2M-subword.vec')
embedding_matrix_fastText = generate_embedding(word_index,word_embedding_fastText,300)
embedding_matrix = embedding_matrix_fastText


#read pretrain word Twitter embeeding Glove
"""
word2vec_output_file = '/home/thindv/ShareTask/Embedding/Glove_twitter_200.word2vec'
word_embedding_glove = KeyedVectors.load_word2vec_format(word2vec_output_file)
embedding_matrix_glove = generate_embedding(word_index,word_embedding_glove,200)
"""

# concatnate two embeddings
#embedding_matrix = np.concatenate((embedding_matrix_glove, embedding_matrix_fastText), axis=1)
PIK4 = "embedding_matrix.dat"
with open(PIK4, "wb") as f:
    pickle.dump(embedding_matrix, f)

"""
PIK4 = "embedding_matrix.dat"
with open(PIK4, "rb") as f:
    embedding_matrix = pickle.load(f)
print(embedding_matrix.shape)
"""
# parameter of model
he_normal_initializer = keras.initializers.he_normal(seed=1234)
glorot_normal_initializer = keras.initializers.glorot_uniform(seed=1995)
recurrent_units = 64
filter_nums = 300 
dropout_rate = 0.2

# calculate class weight
suggest = 0
nonsuggest = 0
for y in document_Y_train:
   xx = np.argmax(y)
   if xx == 1:
       suggest +=1
   else:
       nonsuggest +=1
print(suggest,nonsuggest)
y_integers = np.argmax(document_Y_train, axis=1)
class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
d_class_weights = dict(enumerate(class_weights))



# model
inputs = Input(shape=(maxlen_word, ), dtype='int32', name='word_inputs')
embedding_layer = Embedding(input_vocab_size,embedding_matrix.shape[1],input_length=maxlen_word, weights= [embedding_matrix] ,trainable=True)(inputs)
embedding_layer = SpatialDropout1D(dropout_rate)(embedding_layer)

rnn_1 = Bidirectional(CuDNNLSTM(recurrent_units, return_sequences=True))(embedding_layer)
rnn_2 = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(rnn_1)
embedding_layer = concatenate([rnn_1, rnn_2], axis=2)


conv_1 = Conv1D(filter_nums, 3, kernel_initializer=he_normal_initializer, padding="same", activation="relu")(embedding_layer)
conv_2 = Conv1D(filter_nums, 4, kernel_initializer=he_normal_initializer, padding="same", activation="relu")(embedding_layer)
conv_3 = Conv1D(filter_nums, 5, kernel_initializer=he_normal_initializer, padding="same", activation="relu")(embedding_layer)


maxpool_1 = GlobalMaxPooling1D()(conv_1)
attn_1 = AttentionWeightedAverage()(conv_1)
avg_1 = GlobalAveragePooling1D()(conv_1)


maxpool_2 = GlobalMaxPooling1D()(conv_2)
attn_2 = AttentionWeightedAverage()(conv_2)
avg_2 = GlobalAveragePooling1D()(conv_2)


maxpool_3 = GlobalMaxPooling1D()(conv_3)
attn_3 = AttentionWeightedAverage()(conv_3)
avg_3 = GlobalAveragePooling1D()(conv_3)

v0_col = Concatenate(axis=1)([maxpool_1, maxpool_2, maxpool_3])
v1_col = Concatenate(axis=1)([attn_1, attn_2, attn_3])
v2_col = Concatenate(axis=1)([avg_1, avg_2, avg_3])

merged_tensor = Concatenate(axis=1)([v0_col, v1_col, v2_col])

fc1 = Dense(128,kernel_initializer=glorot_normal_initializer)(merged_tensor)
fc1 = PReLU()(fc1)
fc1 = BatchNormalization()(fc1)
fc1 = Dropout(rate = 0.4,seed = 89)(fc1)

fc2 = Dense(64,kernel_initializer=glorot_normal_initializer)(fc1)
fc2 = PReLU()(fc2)
fc2 = BatchNormalization()(fc2)
fc2 = Dropout(rate = 0.4, seed= 1234)(fc2)

output = Dense(2, activation='sigmoid', kernel_initializer=glorot_normal_initializer)(fc2)


# define optimizer
optimizer = optimizers.SGD(lr=0.01)
model = Model(inputs = inputs, outputs=output)
tensorBoardCallback = TensorBoard(log_dir='./logs', write_graph=True)
#model.compile(loss = lambda y_true, y_pred: focal_loss(y_true, y_pred, 1.6, 2), optimizer=optimizer, metrics=['accuracy'])
model.compile(loss=lambda y_true,y_pred: y_true*K.relu(0.9-y_pred)**2 + 0.5*(1-y_true)*K.relu(y_pred-0.1)**2,optimizer=optimizer, metrics=['accuracy'])
#model.compile(loss="squared_hinge",optimizer=optimizer, metrics=['accuracy'])
ratio = {0:1.0, 1:3.5}
EarlyStop = keras.callbacks.EarlyStopping(monitor='loss', min_delta=1e-4, patience=5, verbose=0, mode='auto')
Reduce = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, mode='auto')
#history = model.fit(totalX, np.array(document_Y_train),batch_size=100,class_weight=ratio,epochs=100,callbacks=[Reduce,EarlyStop,tensorBoardCallback])
history = model.fit(totalX, np.array(document_Y_train),batch_size=100,class_weight=ratio,epochs=100,callbacks=[tensorBoardCallback])

#model.summary()
#model.save('Capsule_Network.h5')


#from keras.models import load_model
#model = load_model('models.h5')


test_path = 'dataset/SubtaskA_EvaluationData.csv'
testData = read_Test_csv(test_path)
document_X_test = list()
document_X_test_raw = list()
test_process = ""
for data in testData:
    document_X_test.append(clean_doc(data[1]))
    test_process += clean_doc(data[1]) + "\n"
    document_X_test_raw.append(data[1])


import operator
textArray_test = np.array(pad_sequences(input_tokenizer.texts_to_sequences(document_X_test), maxlen=maxlen_word))

count = 0
label_list = []
model_list = []
for index2,item in enumerate(textArray_test):
    predicted = model.predict(np.expand_dims(item, axis=0))
    index, value = max(enumerate(predicted[0]), key=operator.itemgetter(1))
    label_list.append(index)
print(len(label_list))

out_path = '/Result/submission.csv'
write_csv(testData, label_list, out_path)

# load dev
test_path = '/dataset/SubtaskA_Trial_Test.csv'
testData = read_Test_csv(test_path)
document_X_test = list()
document_X_test_raw = list()
test_process = ""
for data in testData:
    document_X_test.append(clean_doc(data[1]))
    test_process += clean_doc(data[1]) + "\n"
    document_X_test_raw.append(data[1])



textArray_test = np.array(pad_sequences(input_tokenizer.texts_to_sequences(document_X_test), maxlen=maxlen_word))

count = 0
label_list = []
model_list = []
for index2,item in enumerate(textArray_test):
    predicted = model.predict(np.expand_dims(item, axis=0))
    index, value = max(enumerate(predicted[0]), key=operator.itemgetter(1))
    label_list.append(index)
print(len(label_list))

out_path = '/Result/submission_dev.csv'
write_csv(testData, label_list, out_path)


plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
K.clear_session()
