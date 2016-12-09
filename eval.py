import numpy as np
import csv, sys

def accuracy(pred, actual):
	accuracy = 0.0
	n = 0
	for p, y in zip(pred, actual):
		n += 1
		if p == y:
			accuracy += 1
	return accuracy / n

def mse(pred, actual):
	ms = 0.0
	n = 0
	for p, y in zip(pred, actual):
		n += 1
		ms += (p - y)**2
	return np.sqrt(ms / n)

def ranking(pred, actual):
	'''
	Qin, please complete this section
	'''
	return 0

def evaluate(pred_file):
	'''
	A function that evaluates predicted score against actual.
	The final score is averaged among all essay sets.

	Attributes
	----------
	pred_file: File name of csv file containing all predictions, with the columns:

		1. Set ID of essay
		2. Essay ID
		3. Predicted score
		4. Actual score

		Note: file should be delimited by ',' comma.
	'''
	
	pred = []
	actual = []

	with open(pred_file, "r") as f:
		reader = csv.reader(f, delimiter = ",")
		for line in reader:
			pred.append(int(line[2]))
			actual.append(int(line[3]))

	print("Classification Accuracy: {}".format(accuracy(pred, actual)))
	print("MSE: {}".format(mse(pred, actual)))
	print("Ranking: {}".format(ranking(pred, actual)))

	return True

if __name__ == "__main__":
	evaluate(sys.argv[1])
