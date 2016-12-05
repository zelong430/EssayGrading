import pandas as pd
import numpy as np 
from evaluate import set_range, evaluate

# read in the data
train_data = pd.read_csv("processed_data/train.tsv", delimiter='\t')
pred_y = np.zeros((len(train_data), 3))
pred_y[:, 0] = train_data["essay_set"]
pred_y[:,1] = train_data["essay_id"]
for i in range(1,9):
	pred_y[pred_y[:,0] == i, 2] = np.random.randint(low = set_range[i][0], high = set_range[i][1], size = len(pred_y[pred_y[:,0] == i, 2]))
pred_y = pred_y.astype("int64")
print evaluate(pred_y, against = 0)
