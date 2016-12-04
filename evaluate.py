
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
	The final score is averaged among all possible set of essay.

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

	_against: integer with 3 possible values 0,1,2
		When this argument is provided, the numpy array provided in true_y will be ignore.
		The function will evaluate pred_y against predefined labels store in the "processed_data"
		folder. 

		against = 0: will evaluate pred_y against train.tsv
		against = 1: will evaluate pred_y against val.tsv
		against = 2: will evaluate pred_y against test.tsv
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


def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
	"""
	Returns the confusion matrix between rater's ratings
	"""
	assert(len(rater_a) == len(rater_b))
	if min_rating is None:
		min_rating = min(rater_a + rater_b)
	if max_rating is None:
		max_rating = max(rater_a + rater_b)
	num_ratings = int(max_rating - min_rating + 1)
	conf_mat = [[0 for i in range(num_ratings)]
				for j in range(num_ratings)]
	for a, b in zip(rater_a, rater_b):
		conf_mat[a - min_rating][b - min_rating] += 1
	return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
	"""
	Returns the counts of each type of rating that a rater made
	"""
	if min_rating is None:
		min_rating = min(ratings)
	if max_rating is None:
		max_rating = max(ratings)
	num_ratings = int(max_rating - min_rating + 1)
	hist_ratings = [0 for x in range(num_ratings)]
	for r in ratings:
		hist_ratings[r - min_rating] += 1
	return hist_ratings


def quadratic_weighted_kappa(rater_a, rater_b, min_rating=None, max_rating=None):
	"""
	Calculates the quadratic weighted kappa
	quadratic_weighted_kappa calculates the quadratic weighted kappa
	value, which is a measure of inter-rater agreement between two raters
	that provide discrete numeric ratings.  Potential values range from -1
	(representing complete disagreement) to 1 (representing complete
	agreement).  A kappa value of 0 is expected if all agreement is due to
	chance.
	quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
	each correspond to a list of integer ratings.  These lists must have the
	same length.
	The ratings should be integers, and it is assumed that they contain
	the complete range of possible ratings.
	quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
	is the minimum possible rating, and max_rating is the maximum possible
	rating
	"""
	rater_a = np.array(rater_a, dtype=int)
	rater_b = np.array(rater_b, dtype=int)
	assert(len(rater_a) == len(rater_b))
	if min_rating is None:
		min_rating = min(min(rater_a), min(rater_b))
	if max_rating is None:
		max_rating = max(max(rater_a), max(rater_b))
	conf_mat = confusion_matrix(rater_a, rater_b,
								min_rating, max_rating)
	num_ratings = len(conf_mat)
	num_scored_items = float(len(rater_a))

	hist_rater_a = histogram(rater_a, min_rating, max_rating)
	hist_rater_b = histogram(rater_b, min_rating, max_rating)

	numerator = 0.0
	denominator = 0.0

	for i in range(num_ratings):
		for j in range(num_ratings):
			expected_count = (hist_rater_a[i] * hist_rater_b[j]
							  / num_scored_items)
			d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
			numerator += d * conf_mat[i][j] / num_scored_items
			denominator += d * expected_count / num_scored_items

	return 1.0 - numerator / denominator

if __name__ == "__main__":
	p = read_label(0)
	evaluate(p, against = 0)
