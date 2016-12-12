import numpy as np
import csv, sys
from evaluate import quadratic_weighted_kappa, evaluate


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
        ms += (p - y) ** 2
    return np.sqrt(ms / n)


def ranking(pred_file, ndcg_num = -1):
    '''
    Calculating average NDCG score
    '''
    score_pairs = {i:[] for i in range(1, 9)}
    with open(pred_file, 'r') as f:
        for line in f.readlines():
            fields = line.rstrip().split(',')
            score_pairs[int(fields[0])].append((float(fields[2]), float(fields[3])))
    avgNDCG = 0.0
    cnt = 0
    for i in range(1, 9):
        sp = score_pairs[i]
        if sp == []: continue
        cnt += 1
        if ndcg_num == -1:
            n = len(sp)
        else:
            n = ndcg_num

        sp.sort(key=lambda x:x[0], reverse=True)
        DCG = 0.0
        for k in range(n):
            DCG += sp[k][1]/np.log2(k+2)
            '''
            Choose the following line to make it harder!
            '''
            # DCG += np.exp2(sp[k][1]) / np.log2(k + 2)

        sp.sort(key=lambda x:x[1], reverse=True)
        IDCG = 0.0
        for k in range(n):
            IDCG += sp[k][1]/np.log2(k+2)
            '''
            Choose the following line to make it harder!
            '''
            # IDCG += np.exp2(sp[k][1]) / np.log2(k + 2)

        avgNDCG += DCG/IDCG
        #print '  {0}th class NDCG: {1}'.format(i, DCG/IDCG)

    return avgNDCG/cnt


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
        reader = csv.reader(f, delimiter=",")
        for line in reader:
            pred.append(int(float(line[2])))
            actual.append(int(float(line[3])))

    print("Classification Accuracy: {}".format(accuracy(pred, actual)))
    print("MSE: {}".format(mse(pred, actual)))
    print("Average NDCG: {}".format(ranking(pred_file)))
    print ('Quadratic Weighted Kappa: {}'.format(quadratic_weighted_kappa(pred, actual)))

    return True


if __name__ == "__main__":
    evaluate(sys.argv[1])
