# Automated Essay Grading from Scratch

This is the final project **Automated Essay Grading from Scratch** by Won Lee, Qin Lyu, and Zelong Qiu for the graduate course 6.864 *Advanced Natural Language Processing* at MIT.

## Quickstart

```
python modelX.py
```

where X can be:

* 1: Character-level CNN
* 2: Word-level MLP
* 3: Word-level Bidirectional LSTM
* 4: Ranking Model

## Output Format

All output is in `.csv` format containing the following columns, delimited by comma (`,`).

| Set ID | Essay ID | Predicted Score | Actual Score |
| :----: | :------: | :-------------: | :----------: |
| 1 	 | 1 	    | 8 	      | 8 	     |
| ...	 | ...	    | ...	      | ...	     |

## Evaluation

The output is evaluated using:

```
python eval.py [pred_file].csv
```

where the `pred_file` is the name of the file containing the above output. The evaluation script reports:

* Classification accuracy: Raw classification rate, treating each score as a separate category.
* MSE: Mean-squared error of the predictions (across all essay sets).
* Ranking: