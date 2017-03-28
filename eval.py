import numpy as np
import h5py
from keras.models import load_model
import argparse
import sys
import time
def recall(predy, k):
    num_correct = 0
    num_examples = len(predy)/10
    for i in range(0,len(predy),10):
            preds = np.argsort(-np.array(predy[i:i+10]))
            if 0 in preds[:k]:
                num_correct = num_correct+ 1
    return num_correct/float(num_examples)


def load_data(path):
    f = h5py.File(path, "r")
    x1 = f["x1"][:]
    x2 = f["x2"][:]
    y = f["y"][:]
    f.close()
    return x1, x2, y


def main(arguments):
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_path', help="Path to model",required=True)
    parser.add_argument('--data_path', help="Path to validation data", required=True)
    args = parser.parse_args(arguments)
    model = load_model(args.model_path)
    x1, x2, y = load_data(args.data_path)
    start = time.time()
    predy = model.predict([x1, x2])
    end = time.time()
    print "Speed: ",(end-start)/len(x1),"secs/pair"
    predy = [i[0] for i in predy]
    print "Recall 1 in 10 @ 1:",recall(predy,1)
    print "Recall 1 in 10 @ 2:",recall(predy,2)
    print "Recall 1 in 10 @ 5:",recall(predy,5)
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
