
import numpy as np
import pandas as pd

set_range = {
	1:(2, 12),
	2:(1, 6),
	3:(0,3),
	4:(0,3),
	5:(0,4),
	6:(0,4),
	7:(0,30),
	8:(0,60) 
}

def evaluate(pred_y, true_y = None, against = None):
	""" 
	A function that evaluate the result of predicted rate against true rate.
	The final score is averaged among all essay sets.

	Attributes
	----------
	
	_pred_y: 2D numpy array with 3 columns of predicted score for each essay
		1st column contains the essay_set id of each essay, int
		2nd column contains the essay_id of each essay, int
		3rd column contains the rate for each essay, must be integer, int

		1. number of rows of pred_y must be same with true_y
		2. the essay_set id and essay_id(i.e the first two values) on each row must
			match between pred_y and true_y
		3. the third column must only contain integer values

	_true_y: 2D numpy array with 3 columns of true score for each essay
		1st column contains the essay_set id of each essay, int
		2nd column contains the essay_id of each essay, int
		3rd column contains the rate for each essay, must be integer, int
		
		1. number of rows of pred_y must be same with true_y
		2. the essay_set id and essay_id(i.e the first two values) on each row must
			match between pred_y and true_y
		3. the third column must only contain integer values
		4. if no true_y is provided, the function will evaluate against predefined
			results for train/validate/test set by providing one of the three options
			for the argument against
	"""
	assert ((true_y is not None) or (against is not None))
	if against is not None:
		true_y = read_label(against)
	if pred_y.dtype != 'int64' or true_y.dtype != 'int64':
		raise ValueError("The two numpy arrays must be numpy array with data type int64")
	if np.all(pred_y[:,0] == true_y[:,0]) != True or np.all(pred_y[:,1] == true_y[:,1]) != True:
		raise ValueError('The first two columns of pred_y and true_y must match, please check the essay_id and essay_set id')
	list_kappa = np.zeros(8)
	for i in range(1, 9):
		list_kappa[i-1] = quadratic_weighted_kappa(pred_y[pred_y[:,0] == i, 2], true_y[true_y[:,0] == i, 2],\
													min_rating=set_range[i][0], max_rating=set_range[i][1]
													)
	list_kappa[list_kappa > 0.999] = 0.999
	z = 0.5 * np.log((1 + list_kappa)/(1 - list_kappa))	
	z = z.mean()
	averaged_kappa = (np.exp(2*z) - 1)/(np.exp(2*z) + 1)
	return averaged_kappa

def read_label(file):
	if file == 0:
		data = pd.read_csv("processed_data/train.tsv", delimiter='\t')
	elif file == 1:
		data = pd.read_csv("processed_data/val.tsv", delimiter='\t')
	elif file == 2:
		data = pd.read_csv("processed_data/test.tsv", delimiter='\t')
	else:
		raise ValueError('parameter against, if provided, could only be 0,1,3')

	result = np.zeros((len(data), 3))
	result[:,0] = data["essay_set"]
	result[:,1] = data["essay_id"]
	result[:,2] = data["domain1_score"]
	result = result.astype("int64")
	return result

if __name__ == "__main__":
	p = read_label(0)
	evaluate(p, against = 0)
