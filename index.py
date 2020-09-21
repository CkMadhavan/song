import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import string
import re
from flask import Flask

from flask_cors import CORS, cross_origin
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

keras.backend.clear_session()
    
t_inp = pickle.load(open('t_lyrics' , "rb"))
t_oup = t_inp
t = t_inp

Xlen = 11
Ylen = 50

Xvocab = len(t.word_index) + 1
Yvocab = len(t.word_index) + 1

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

units = 32

inp2 = keras.layers.Input(shape=(Xlen, ))
enc1 = keras.layers.Embedding(Xvocab , 100,input_length = Xlen)(inp2)
enc1 = keras.layers.Bidirectional(keras.layers.LSTM(units , return_sequences = True))(enc1)
enc1 = keras.layers.Dropout(0.4)(enc1)
enc1 = keras.layers.BatchNormalization()(enc1)

attention = keras.layers.Dense(1, activation='tanh')(enc1)
attention = keras.layers.Flatten()(attention)
attention = keras.layers.Activation('softmax')(attention)
attention = keras.layers.RepeatVector(2*units)(attention)
attention = keras.layers.Permute([2, 1])(attention)

sent_representation = keras.layers.multiply([enc1, attention])
sent_representation1 = keras.layers.Lambda(lambda xin: keras.backend.sum(xin, axis=-2), output_shape=(2*units,))(sent_representation)

inp3 = keras.layers.Input(shape=(Ylen, ))
enc2 = keras.layers.Embedding(Yvocab , 100, input_length = Ylen)(inp3)
enc2 = keras.layers.Bidirectional(keras.layers.LSTM(units , return_sequences = True))(enc2)
enc2 = keras.layers.Dropout(0.4)(enc2)
enc2 = keras.layers.BatchNormalization()(enc2)

attention = keras.layers.Dense(1, activation='tanh')(enc2)
attention = keras.layers.Flatten()(attention)
attention = keras.layers.Activation('softmax')(attention)
attention = keras.layers.RepeatVector(2*units)(attention)
attention = keras.layers.Permute([2, 1])(attention)

sent_representation = keras.layers.multiply([enc2, attention])
sent_representation2 = keras.layers.Lambda(lambda xin: keras.backend.sum(xin, axis=-2), output_shape=(2*units,))(sent_representation)

decoder = keras.layers.add([sent_representation1,sent_representation2])
out = keras.layers.Dense(Yvocab , activation='softmax')(decoder)

model = keras.models.Model(inputs = [inp2,inp3] , outputs = out)

model.compile(optimizer = keras.optimizers.Adam(lr = 0.001) , loss = 'categorical_crossentropy' , metrics = ['acc'])
model.load_weights('lyrics-eng.hdf5')

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def prediction(model , inp_que , inp = '' , totlen = 50):

  que = pad_sequences(t.texts_to_sequences([inp_que]) , maxlen = Xlen , padding='pre' , truncating = 'pre')
  if inp == '':
    text = 'startseq'
  else:
    text = 'startseq ' + inp
  for i in range(totlen):
    ans = pad_sequences(t.texts_to_sequences([text]) , maxlen = Ylen , padding='pre' , truncating = 'pre')
    y_pred = t.sequences_to_texts([[np.argsort(model.predict([que.reshape(1,Xlen) , ans.reshape(1,Ylen)]))[0][-np.random.randint(1,2)]]])[0]

    if y_pred == 'endseq':
      if text == 'startseq':
        text += ' ' + t.sequences_to_texts([[np.argsort(model.predict([que.reshape(1,Xlen) , ans.reshape(1,Ylen)]))[0][-2]]])[0]
      else:
        text += ' ' + y_pred
    elif y_pred == 'nexteee':
      if text[-7:] == 'nexteee':
        text += ' ' + t.sequences_to_texts([[np.argsort(model.predict([que.reshape(1,Xlen) , ans.reshape(1,Ylen)]))[0][-2]]])[0]
      else:
        text += ' ' + y_pred
    else:
      text += ' ' + y_pred

    if y_pred == 'endseq':
      break

  return text

def clean(line):
    
    re_print = re.compile('[^%s]' % re.escape(string.printable))
    table = str.maketrans('' , '' , string.punctuation)
    
    line = line.lower()
    #line = normalize('NFD' , (line)).encode('ascii' , 'ignore')
    
    line = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", line)
    
    line = line.split()
    line = [word.translate(table) for word in line]
    #line = [re_print.sub('' , w) for w in line]
    #line = [word for word in line if word.isalpha()]
    line = ' '.join(line)
    line = 'startseq ' + line + ' endseq'
        
    return line
    
@app.route('/<process>')
#@cross_origin()
def index(process):
    process = clean(process)
    return prediction(model , process)[9:-7]

if __name__ == "__main__":
    app.run()
