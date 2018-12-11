import nltk
from nltk import tokenize
import glob, sys, os, pickle
import itertools
import numpy as np

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

"""
Usage:

python2 ngram_overlap.py results_dir data_path
"""

def make_ngrams(sentence, order):
  return zip(*[sentence[i:] for i in range(order)])

def ngram_overlap(summary, doc, order):
    sum_ng = set()
    doc_ng = set()
    sum_ng = set(make_ngrams(tokenize.word_tokenize(summary), order))
    doc_ng = set(make_ngrams(tokenize.word_tokenize(doc), order))
    novel = sum_ng.difference(doc_ng)
    return len(novel) / float(len(sum_ng))

def sentence_overlap(summary, doc):
    sum_s = set(tokenize.sent_tokenize(summary))
    doc_s = set(tokenize.sent_tokenize(doc))
    novel = sum_s.difference(doc_s)
    return len(novel) / float(len(sum_s))

def restore_order(batch_order):
    """
    reconstruct index from batch_order.

    Assumes original indices are in numerical order.
    """
    batch_size = len(batch_order[0])
    return [batch_order[i] + batch_size*i for i in xrange(len(batch_order))]

def main(folder, data_path):
    # Load test data
    with open(data_path, 'r') as f:
        data = pickle.load(f)

    # Load summaries
    sum_files = glob.glob(os.path.join(folder, 'beam_summary/*'))
    sum_files = {int(os.path.basename(f)): f for f in sum_files}

    # Load file order
    order_file = os.path.join(folder, 'test_batch_order.pkl')
    with open(order_file, 'r') as f:
        order = pickle.load(f)
    order = restore_order(order)
    order = list(itertools.chain.from_iterable([list(o) for o in order]))

    # Calculate novel ngrams and sentences for each summary
    ng_overlaps = {i:0 for i in range(1, 5)}
    sent_overlaps = 0
    for i in xrange(len(sum_files)):
        with open(sum_files[i], 'r') as f:
            summ = f.readlines()[-1].split(' ', 1)[1].decode('utf-8')

        source = data[order[i]][0][1].decode('utf-8')

        for o in range(1, 5):
            no = ngram_overlap(summ, source, o)
            ng_overlaps[o] += no
        so = sentence_overlap(summ, source)
        sent_overlaps += so

    # Print results  
    for i in range(1, 5):
        mean_i = ng_overlaps[i] / float(len(sum_files))
        print('novel {}-grams:\t{:.4f}'.format(i, mean_i))
    mean_s = sent_overlaps / float(len(sum_files))
    print('novel sentences:{:.4f}'.format(mean_s))

if __name__ == "__main__":
    folder = sys.argv[1]
    if len(sys.argv) == 3:
        data_path = sys.argv[2]
    else:
        data_path = '../deepmind/test_set/test_500.pkl'
    main(folder, data_path)
