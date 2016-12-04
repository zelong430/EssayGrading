import re
datafiles = ['../processed_data/train.tsv','../processed_data/val.tsv',
            '../processed_data/test.tsv']
'''
Schema:
essay_id essay_set essay rater1_domain1 rater2_domain1 domain1_score

The entitities identified by NER are:
    "PERSON", "ORGANIZATION", "LOCATION", "DATE", "TIME", "MONEY", "PERCENT"
'''
def clean_data(text, keep_period = False):
    text = text.lower()

    NER_pat = re.compile("(@[a-z]+)[1-9]+")
    text = NER_pat.sub('\\1', text)

    NUM_pat = re.compile("(\d+)\S*")
    text = NUM_pat.sub('@number', text)

    if keep_period:
        text = re.sub('[^a-zA-Z0-9.@]', ' ', text)
    else:
        text = re.sub('[^a-zA-Z0-9@]', ' ', text)
    return text

word_list = []
for file in datafiles:
    with open(file) as f:
        for line in f.readlines():
            # essay = NER_pat.sub('\\1', line.rstrip().split('\t')[2])
            # essay = NUM_pat.sub('NUMBER', essay)
            # word_list += re.split('\W', essay.lower())
            # word_list += essay.lower().split()
            word_list += clean_data(line.rstrip().split('\t')[2]).split()

words = sorted(list(set(word_list)))
for word in words: print(word)
print('total words:', len(words))
word2idx = dict((w, i) for i, w in enumerate(words))
idx2word = dict((i, w) for i, w in enumerate(words))
