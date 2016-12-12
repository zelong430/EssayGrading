import pandas as pd
import numpy as np 
from evaluate import set_range, evaluate

# read in the data
train_data = pd.read_csv("processed_data/train.tsv", delimiter='\t')

pred_y = np.zeros((len(train_data), 4))
pred_y[:, 0] = train_data["essay_set"]
pred_y[:,1] = train_data["essay_id"]
for i in range(1,9):
	pred_y[pred_y[:,0] == i, 2] = np.random.randint(low = set_range[i][0], high = set_range[i][1], size = len(pred_y[pred_y[:,0] == i, 2]))
pred_y[:, 3] = train_data["domain1_score"]
pred_y = pred_y.astype('int')

# result = train_data["essay_set"]
# result['essay_id'] = train_data["essay_id"].values
# result['prediction'] =  pred_y[:, 2]
# result['actual'] = train_data["domain1_score"].values
# result.to_csv('test.csv', sep=',', index = False)

np.savetxt("sample_predictions_train.csv", pred_y, delimiter=",", fmt='%d')
