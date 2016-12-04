import re
import pickle
import numpy as np
from gensim.models import word2vec

datafiles = ['../processed_data/train.tsv', '../processed_data/val.tsv',
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

word2emb = word2vec.Word2Vec.load_word2vec_format('text8-vector.bin', binary=True)
for file, sfile in zip(datafiles, ['train.pkl', 'val.pkl', 'test.pkl']):
    docset = {i + 1: [] for i in range(8)}
    with open(file) as f:
        skip = 1
        for line in f.readlines():
            if skip == 1:
                skip = 0
                continue
            # word_list += clean_data(line.rstrip().split('\t')[2]).split()
            fields = line.rstrip().split('\t')
            sentence_list = clean_data(fields[2], True).split('.')
            cur_doc = []
            for sentence in sentence_list:
                sentence_emb = np.zeros(200)
                store = 0
                for word in sentence.split():
                    try:
                        sentence_emb += word2emb[word]
                        store = 1
                    except:
                        pass
                if store == 1:
                    cur_doc.append(sentence_emb)

            docset[int(fields[1])].append((np.reshape(cur_doc, (len(cur_doc), 200)), int(fields[-1])))

    with open(sfile, 'wb') as g:
        pickle.dump(docset, g)
