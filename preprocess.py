import csv
import sys
import argparse
import numpy as np
import h5py
import string
import re
from collections import defaultdict


def tokenize(sent):
    sent = re.sub('[%s]' % string.punctuation, ' ', sent)
    sent = string.lower(sent.strip()).split()
    sent = [i for i in sent if len(i)!=0]
    return sent


def write_vocab(vocab, outfile, chars=0):
    out = open(outfile, "w")
    items = [(v, k) for k, v in vocab.iteritems()]
    items.sort()
    for v, k in items:
            print >>out, k, v
    out.close()


def pad(ls, length, symbol):
    if len(ls) >= length:
        return ls[(len(ls)-length):len(ls)]
    return [symbol] * (length -len(ls))+ls

def pad2(ls, length, symbol):
    if len(ls) >= length:
        return ls[0:length]
    return [symbol] * (length -len(ls))+ls

def prune_vocab(vocab,k):
    d = {"<blank>":0,"<unk>": 1}
    vocab_list = [(word, count) for word, count in vocab.iteritems()]
    vocab_list.sort(key=lambda x: x[1], reverse=True)
    k = min(k-2, len(vocab_list))
    pruned_vocab = [pair[0] for pair in vocab_list[:k]]
    for word in pruned_vocab:
        if word not in d:
            d[word] = len(d)
    return d


def make_vocab(trainfile):
    vocab = defaultdict(int)
    with open(trainfile) as csvfile:
            filereader = csv.reader(csvfile,delimiter=',')
            next(filereader, None)
            for row in filereader:
                qry = tokenize(row[0])
                rsp = tokenize(row[1])
                for w1 in qry:
                    vocab[w1]+=1
                for w2 in rsp:
                    vocab[w2]+=1
            return vocab

#convert training data
def convert(datafile,vocab,seql1,seql2):
    qlist = []
    rlist = []
    y = []
    count = 0
    with open(datafile) as csvfile:
            filereader = csv.reader(csvfile,delimiter=',')
            next(filereader, None)
            for row in filereader:
                count = count +1
                if count%100000==0:
                    print count," lines processed"
                    break
                if len(row)!=3:
                    continue
                qry = tokenize(row[0])
                rsp = tokenize(row[1])
                for i in xrange(0,len(qry)):
                    qry[i] = qry[i] if qry[i] in vocab else "<unk>"
                for i in xrange(0,len(rsp)):
                    rsp[i] = rsp[i] if rsp[i] in vocab else "<unk>"
                if len(qry) < 1 or len(rsp) < 1:
                    continue
                qryint = [vocab[i] for i in qry]
                qryint = pad(qryint,seql1,0)
                rspint = [vocab[i] for i in rsp]
                rspint = pad(rspint,seql2,0)
                qlist.append(qryint)
                rlist.append(rspint)
                y.append(int(row[2]))
    qarray = np.array(qlist)
    rarray = np.array(rlist)
    yarray = np.array(y)
    return qarray, rarray, yarray

#convert validation data
def convert2(datafile,vocab,seql1,seql2):
    qlist = []
    rlist = []
    y = []
    count = 0
    with open(datafile) as csvfile:
            filereader = csv.reader(csvfile,delimiter=',')
            next(filereader, None)
            for row in filereader:
                count = count +1
                if count%1000==0:
                    print count," lines processed"
                    break
                if len(row)<11:
                    continue
                qry = tokenize(row[0])
                for i in xrange(0, len(qry)):
                    qry[i] = qry[i] if qry[i] in vocab else "<unk>"
                qryint = [vocab[i] for i in qry]
                qryint = pad(qryint,seql1,0)
                for i in range(1,11):
                    rsp = tokenize(row[i])
                    for i in xrange(0,len(rsp)):
                        rsp[i] = rsp[i] if rsp[i] in vocab else "<unk>"
                    if len(qry) < 1 or len(rsp) < 1:
                        continue
                    rspint = [vocab[i] for i in rsp]
                    rspint = pad2(rspint,seql2,0)
                    qlist.append(qryint)
                    rlist.append(rspint)
                    if i==1:
                        y.append(1)
                    else:
                        y.append(0)
    qarray = np.array(qlist)
    rarray = np.array(rlist)
    yarray = np.array(y)
    return qarray, rarray, yarray

def get_data(args):
    print "Generating vocabulary ...\n"
    vocab = make_vocab(args.trainfile)
    dic = prune_vocab(vocab, args.vocabsize)
    print "Saving vocabulary to "+args.outputfile+"_vocab.txt ...\n"
    write_vocab(dic, args.outputfile + "_vocab.txt")
    print "Converting training data ... \n"
    train_q,train_r,train_y=convert(args.trainfile,dic,args.maxseqc,args.maxsequ)
    print "Saving training data ..."
    f = h5py.File(args.outputfile+"-train.hdf5", "w")
    f["train_q"]=train_q
    f["train_r"]=train_r
    f["train_y"]=train_y
    f["maxseqc"]=np.array([args.maxseqc])
    f["maxsequ"] = np.array([args.maxsequ])
    f["vocabsize"]=np.array([min(args.vocabsize,len(dic))])
    f.close()
    print "Converting validation data ... \n"
    valid_q,valid_r,valid_y=convert2(args.validfile,dic,args.maxseqc,args.maxsequ)
    f = h5py.File(args.outputfile+"-valid.hdf5", "w")
    f["valid_q"]=valid_q
    f["valid_r"]=valid_r
    f["valid_y"]=valid_y
    f.close()

def main(arguments):
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--vocabsize', help="Size of vocabulary, constructed by taking the top X most frequent words. Rest are replaced with special UNK tokens.",type=int, default=70000)
    parser.add_argument('--trainfile', help="Path to training data", required=True)
    parser.add_argument('--validfile', help="Path to validation data", required=True)
    parser.add_argument('--maxseqc', help="Maximum sequence length of context. Sequences longer than this are truncated.", type=int, default=70)
    parser.add_argument('--maxsequ', help="Maximum sequence length of utterance. Sequences longer than this are truncated.", type=int, default=50)
    parser.add_argument('--outputfile', help="Prefix of the output file names. ", type=str, required=True)
    args = parser.parse_args(arguments)
    get_data(args)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))


