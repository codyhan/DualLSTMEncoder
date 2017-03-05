import numpy as np
import h5py
from keras.models import load_model

def recall(predy, k):
    num_correct = 0
    num_examples = len(predy)/10
    for i in range(0,len(predy),10):
            preds = np.argsort(-np.array(predy[i:i+10]))
            if 0 in preds[:k]:
                num_correct = num_correct+ 1
    return num_correct/float(num_examples)


def load_valid(path):
    f = h5py.File(path, "r")
    x1 = f["valid_q"][:]
    x2 = f["valid_r"][:]
    y = f["valid_y"][:]
    f.close()
    return x1, x2, y

model = load_model("./cc/m_05.hdf5")
x1,x2,y=load_valid("./ubun-valid.hdf5")
predy = model.predict([x1,x2])
predy = [i[0] for i in predy]
print recall(predy,1)
print recall(predy,2)
print recall(predy,5)