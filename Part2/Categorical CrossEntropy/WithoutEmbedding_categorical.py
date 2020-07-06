import pandas as pd
from pandas import crosstab
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential, save_model, load_model
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Flatten, MaxPooling1D, Convolution1D, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import xlsxwriter
import tensorflow as tf
from keras import backend as K

# import data
train = pd.read_csv("D:\\Priya\\Work3\\Test\\cleaned_hm.csv")

# Lets one-hot encode the labels
labels=train.predicted_category.unique()
dic={}
for i,labels in enumerate(labels):
    dic[labels]=i
labels=train.predicted_category.apply(lambda x:dic[x])

val=train.sample(frac=0.2,random_state=200)
train=train.drop(val.index)
NUM_WORDS=20000 # if set, tokenization will be restricted to the top num_words most common words in the dataset).
tokenizer = Tokenizer(num_words=NUM_WORDS,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'',
                      lower=True)

# we need to fit the tokenizer on our text data in order to get the tokens
texts=train.cleaned_hm
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index
vocab_size = len(tokenizer.word_index) + 1

#hmid,cleaned_hm values taken for test sets using val
hmidValues=val.hmid
cleanedhmValues=val.cleaned_hm
# print the test set values in test_set.xlsx
dftest=pd.DataFrame({'hmid':hmidValues,'cleaned_hm':cleanedhmValues})
dftest.to_excel("test_set_categorical.xlsx",sheet_name='sheet1',index=False)

sequences_train = tokenizer.texts_to_sequences(texts) # converts the text to numbers essentially
sequences_valid=tokenizer.texts_to_sequences(val.cleaned_hm)
word_index = tokenizer.word_index

# set the sequence length of the text to speed up training and prevent overfitting.
seq_len = 500
X_train = pad_sequences(sequences_train,maxlen=seq_len, value=0)
X_val = pad_sequences(sequences_valid,maxlen=seq_len, value=0)

y_train =train.predicted_category.apply(lambda x:dic[x])
y_train = to_categorical(np.asarray(labels[train.index]))
y_val =val.predicted_category.apply(lambda x:dic[x])
y_val = to_categorical(np.asarray(labels[y_val.index]))


# Without pretrained embedding, we just initalize the matrixs as:
EMBEDDING_DIM=300
vocabulary_size=min(len(word_index)+1,NUM_WORDS)

embedding_layer = Embedding(vocabulary_size,
                            EMBEDDING_DIM)

# Use a sequential setup
model = Sequential()
e = Embedding(vocab_size, 100, input_length=seq_len,trainable=True)

# Use 1 Convolution Kernal
model.add(e)
model.add(Dropout(0.2))
model.add(Convolution1D(64, 5, padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.7))
model.add(Dense(7, activation='sigmoid'))  # 7 targets, each done as a logistic

# compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

# setup checkpoint
file_path="D:\\Priya\\Work3\\dump\\weights_base.CovNet.hdf5"
checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='min')
early = EarlyStopping(monitor="val_acc", mode="min", patience=20)
callbacks_list = [checkpoint,early] #early
# fit the model
model.fit(X_train, y_train, batch_size=64, epochs=5, validation_split=0.2, callbacks=callbacks_list, verbose=1)

#predict calculate for test sets using sequential model
sequentialpredict = model.predict(X_val, batch_size=64, verbose=1)
classes = np.argmax(sequentialpredict, axis = 1)

# recall method
def recall_m(y_val, sequentialpredict):
    true_positives = K.sum(K.round(K.clip(y_val * sequentialpredict, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_val, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

#precision method
def precision_m(y_val, sequentialpredict):
    true_positives = K.sum(K.round(K.clip(y_val * sequentialpredict, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(sequentialpredict, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

#f1 method
def f1_m(y_val, sequentialpredict):
    precision = precision_m(y_val, sequentialpredict)
    recall = recall_m(y_val, sequentialpredict)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# test set predict values are printed in test_predict.xlsx
dfF = pd.DataFrame({'WithoutEmbedding_Predict':classes})
dfF = dfF.fillna(1)
dfF = dfF.astype(int)
dfF.replace({0:'affection', 1:'exercise', 2:'bonding', 3:'leisure', 4:'achievement' , 5:'enjoy_the_moment', 6:'nature'}, inplace=True)
dfF.to_excel("test_predict_categorical.xlsx",sheet_name='sheet1',index=False)
# concat the hmid,cleaned_hm,predicted_category values to 'WithoutEmbeddingConsolidate.xlsx' for test sets
df1=pd.read_excel('test_set_categorical.xlsx')
df2=pd.read_excel('test_predict_categorical.xlsx')
False_data = pd.DataFrame()
False_data=pd.concat([df1,df2],axis=1)
False_data.sort_values(["hmid", "cleaned_hm" ,"WithoutEmbedding_Predict"], axis=0,
                 ascending=True, inplace=True)
False_data.to_excel("WithoutEmbeddingConsolidate_categorical.xlsx",index=False)

loss,accuracy=model.evaluate(X_val,y_val,verbose=1)
print('[INFO] Evaluate method categorical_crossentropy Without Embedding accuracy: %f' % accuracy)

print('[INFO] K-mean categorical_crossentropy Without Embedding accuracy')
accurac = K.cast(K.equal(K.argmax(y_val, axis=-1),K.argmax(sequentialpredict, axis=-1)),K.floatx())
print(K.mean(accurac))

precision = precision_m(y_val, sequentialpredict)
recall = recall_m(y_val, sequentialpredict)
f1_score = f1_m(y_val, sequentialpredict)
print('[INFO] categorical_crossentropy Without Embedding f1_score: %f' % f1_score)
print('[INFO] categorical_crossentropy Without Embedding precision: %f' % precision)
print('[INFO] categorical_crossentropy Without Embedding recall: %f' % recall)
