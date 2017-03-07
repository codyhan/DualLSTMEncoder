from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop
import argparse
import numpy as np
from keras.layers import Merge
import sys
import h5py
from keras.regularizers import l2
from keras import callbacks
from keras.models import load_model
def train(args):
    print "Loading data ..."
    x1, x2, y, vocab_size, maxseqc, maxsequ = load_train(args.data_file)
    model1 = Sequential()
    model2 = Sequential()
    if not args.pre_word_vec=="":
        print "Loading pre-trained word embeddings ..."
        prewordemb, vec_size = load_wordemb(args.pre_word_vec)
        wordict = load_dict(args.dic_file)
        emb_matrix = np.zeros((vocab_size, vec_size))
        for i, word in wordict.items():
            word_vec = prewordemb.get(word)
            if word_vec is not None:
                emb_matrix[i] = word_vec
            else:
                emb_matrix[i] = np.random.uniform(size=(1, vec_size))*0.02-0.01
        del prewordemb
        model1.add(Embedding(vocab_size,args.word_vec_size,weights=[emb_matrix],mask_zero=True,input_length=maxseqc,trainable=(not args.fix_word_vec),W_regularizer=l2(args.l2)))
        model2.add(Embedding(vocab_size,args.word_vec_size,weights=[emb_matrix],mask_zero=True,input_length=maxsequ,trainable=(not args.fix_word_vec),W_regularizer=l2(args.l2)))
    else:
        model1.add(Embedding(vocab_size,args.word_vec_size,mask_zero=True,input_length=maxseqc,trainable=(not args.fix_word_vec),W_regularizer=l2(args.l2)))
        model2.add(Embedding(vocab_size,args.word_vec_size,mask_zero=True,input_length=maxsequ,trainable=(not args.fix_word_vec),W_regularizer=l2(args.l2)))

    if args.num_layers==1:
        model1.add(LSTM(output_dim=args.rnn_size, dropout_W=args.dropout_lstm, dropout_U=args.dropout_lstm,W_regularizer=l2(args.l2),U_regularizer=l2(args.l2)))
        model2.add(LSTM(output_dim=args.rnn_size, dropout_W=args.dropout_lstm, dropout_U=args.dropout_lstm,W_regularizer=l2(args.l2),U_regularizer=l2(args.l2)))
    else:
        for i in range(0,args.num_layers-1):
            model1.add(LSTM(output_dim=args.rnn_size,return_sequences=True,dropout_W=args.dropout_lstm,dropout_U=args.dropout_lstm,W_regularizer=l2(args.l2),U_regularizer=l2(args.l2)))
            model2.add(LSTM(output_dim=args.rnn_size,return_sequences=True,dropout_W=args.dropout_lstm,dropout_U=args.dropout_lstm,W_regularizer=l2(args.l2),U_regularizer=l2(args.l2)))
        model1.add(LSTM(output_dim=args.rnn_size, dropout_W=args.dropout_lstm, dropout_U=args.dropout_lstm,W_regularizer=l2(args.l2),U_regularizer=l2(args.l2)))
        model2.add(LSTM(output_dim=args.rnn_size, dropout_W=args.dropout_lstm, dropout_U=args.dropout_lstm,W_regularizer=l2(args.l2),U_regularizer=l2(args.l2)))
    model1.add(Dense(args.rnn_size,W_regularizer=l2(args.l2)))
    model1.add(Dropout(args.dropout_mlp))
    merged_model = Sequential()
    merged_model.add(Merge([model1,model2],mode="dot"))
    merged_model.add(Activation('sigmoid'))

    opt = RMSprop(lr=args.lr)
    print "Start training ..."
    merged_model.compile(loss=args.loss, optimizer=opt,metrics=[args.metrics])
    merged_model.fit([x1,x2], y,validation_split=0.001, batch_size=args.batch_size, nb_epoch=args.epochs,callbacks=[callbacks.ModelCheckpoint(args.save_path+"/m_{epoch:02d}.hdf5"),callbacks.TensorBoard(log_dir=args.log_path, histogram_freq=1)])
    merged_model.save(filepath=args.save_path+"/final_model.hdf5")


def train_fr(args):
    x1, x2, y, vocab_size, maxseqc, maxsequ = load_train(args.data_file)
    model = load_model(args.train_from)
    model.fit([x1,x2], y,validation_split=0.001, batch_size=args.batch_size, nb_epoch=args.epochs,callbacks=[callbacks.ModelCheckpoint(args.save_path+"/m_{epoch:02d}.hdf5"),callbacks.TensorBoard(log_dir=args.log_path, histogram_freq=1)])
    model.save(filepath=args.save_path + "/final_model.hdf5")

def load_train(path):
    f = h5py.File(path, "r")
    x1 = f["x1"][:]
    x2 = f["x2"][:]
    y = f["y"][:]
    vocab_size = f["vocabsize"][:][0]
    maxseqc = f["maxseqc"][:][0]
    maxsequ = f["maxsequ"][:][0]
    f.close()
    return x1, x2, y, vocab_size, maxseqc, maxsequ


def load_wordemb(path):
    emb_dict = {}
    f = open(path)
    for line in f:
        values = line.strip().split()
        word = values[0]
        vec = np.asarray(values[1:], dtype='float32')
        emb_dict[word] = vec
    vec_size =len(vec)
    f.close()
    return emb_dict,vec_size

def load_dict(path):
    id2word = {}
    f = open(path)
    for line in f:
        line = line.strip().split()
        word = line[0]
        id = int(line[1])
        id2word[id] = word
    f.close()
    return id2word



def main(arguments):
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_file', help="Path to load training data file", required=True)
    parser.add_argument('--save_path', help="Path to save checking points", required=True)
    parser.add_argument('--log_path', help="Path to save tensorboard logs")
    parser.add_argument('--train_from',help="Train from a save point",default="")
    parser.add_argument('--pre_word_vec',help="If a valid path specified, pre-trained word embeddings will be loaded",type=str,default="")
    parser.add_argument('--dic_file', help="If use pre-trained word embeddings, vocabulary file should also be provided",type=str, default="")
    parser.add_argument('--fix_word_vec',help="If true, word embeddings will be fixed once initialized",type=bool,default=False)

    parser.add_argument('--lr', help="Learning rate", type=float, default=0.001)
    parser.add_argument('--word_vec_size', help="Word embedding size",type=int, default=100)
    parser.add_argument('--rnn_size', help="Size of LSTM hidden states",type=int,default=200)
    parser.add_argument('--num_layers', help="Number of LSTM layers", type=int, default=1)
    parser.add_argument('--dropout_lstm', help="Dropout rate for LSTM layer", type=float, default=0.2)
    parser.add_argument('--l2', help="l2 regularizer", type=float, default=0.001)
    parser.add_argument('--dropout_mlp', help="Dropout rate for MLP layer", type=float, default=0.2)
    parser.add_argument('--loss',help="Loss function",choices=['binary_crossentropy','cosine_proximity'],default='binary_crossentropy')
    parser.add_argument('--metrics',help="Evaluation metrics",choices=['fmeasure', 'recall', 'precision','accuracy'],default='fmeasure')

    parser.add_argument('--batch_size',help="Batch size",type=int,default=128)
    parser.add_argument('--epochs',help="Number of epochs",type=int,default=10)
    args = parser.parse_args(arguments)
    if args.train_from=="":
        train(args)
    else:
        print "train from", args.train_from
        train_fr(args)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
