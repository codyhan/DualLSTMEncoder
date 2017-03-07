#The Dual LSTM Encoder for Retrieval-Based Chatbot(Keras Implementation)

The model is from [The Ubuntu Dialogue Corpus: A Large Dataset for Research in Unstructured Multi-Turn Dialogue Systems](https://arxiv.org/pdf/1506.08909.pdf). The big difference between my implementation and the model from the cited paper is that I build two LSTMs for context and utterance seperately whereas in the cited paper the context and the utterance share one  LSTM module.
##Dependicies
###Python
* h5py
* keras
* tensorflow
* numpy
##Quickstart
We save the Ubuntu data in ./data folder. Data can be download from [here](https://drive.google.com/open?id=0B_bZck-ksdkpVEtVc1R6Y01HMWM). First we run the following code to preprocess data.
```
python preprocessing.py --trainfile ./data/train.csv --validfile ./data/valid.csv --testfile ./data/test.csv --outputfile ubun
```
This step will generate three hdf5 files for train.csv, valid.csv, and test.csv respectively. The file names will be ubun-train.hdf5, ubun-valid.csv, ubun-test.csv. More options with explainations can be found in preprocess.py

Now we can run the model with default configurations.
```
python train.py --data_file ubun-train.hdf5 --save_path ./checkpoints --log_path ./logs 
```
This step will save a checkpoint after each epoch in ./checkpoints folder. It will also save logs for tensorboard in ./logs folder. More options with explainations can be found in train.py

After finishing training, we can check performance plots by activating tensorboard.
```
tensorboard --logdir ./logs
```

The data structure for valid.csv is a bit different from train.csv. Therefore I write a seperate script to evaluate validation data using Recall@k metrics. 

To evaluate validation data and test data, we run
```
python eval.py --model_path ./checkpoints/finalmodel.hdf5 --data_path ./ubun-valid.hdf5
python eval.py --model_path ./checkpoints/finalmodel.hdf5 --data_path ./ubun-test.hdf5
```
With the default setting, the print-out for validation data is 
```
Recall 1 in 10 @ 1:
Recall 1 in 10 @ 2:
Recall 1 in 10 @ 5:
```
Check the test data yourself :D
