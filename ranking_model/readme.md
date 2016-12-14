#CNN/LSTM based Model with Pairwise Ranking Loss
The scripts should be run in the following order:

1. preprocess.py

This file reads in all the data, and transform them to pickle files that can be readily used at later stage.
Word embeddings are provided by [GloVe](http://nlp.stanford.edu/projects/glove/).
The embedding file should be named "glove.840B.300d.txt" and be put into the same folder, otherwise it will reads in "word2emb.pkl" instead.

This file will produce "train.pkl" and "val.pkl" and "word2emb.pkl", which is used at training.

2. train.py

This file is used for training. Pairwise ranking loss is used. You can choose between CNN and LSTM models.

3. predict.py

This file reads in "test.tsv" and "score_model.h5" (trained model), and predicts the result for each sample in "test.tsv".
The result is stored in "result.csv".
