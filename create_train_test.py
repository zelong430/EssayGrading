import pandas as pd
import numpy as np

def split_train_test(df, essay_set_id, size = 0.8):
    data = df[df["essay_set"] == essay_set_id]
    mask = np.random.rand(len(data))
    train_mask = mask < size
    validate_mask = np.logical_and((size < mask), (mask < (1 + size) / 2))
    test_mask = np.logical_and((1 + size) / 2 <= mask, mask < 1)

    train_df = data[train_mask]
    validate_df = data[validate_mask]
    test_df = data[test_mask]

    return train_df, validate_df, test_df

def main():
    data = pd.read_csv('data/training_set_rel3.tsv',delimiter='\t')
    data = data.dropna(axis=1)

    train = []
    val = []
    test = []
    for i in range(9):
        a,b,c = split_train_test(data, i, size = 0.8)
        train.append(a)
        val.append(b)
        test.append(c)
    train = pd.concat(train) 
    val = pd.concat(val)
    test = pd.concat(test)

    train.to_csv('train.tsv', sep='\t', index = False)
    val.to_csv('val.tsv', sep='\t')
    test.to_csv('test.tsv', sep='\t')


if __name__ == "__main__":
    main()
