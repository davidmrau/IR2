import nltk
from nltk import tokenize
import glob, sys, os, pickle
import itertools
import numpy as np
import collections

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

"""
Usage:

python2 ngram_overlap_self.py results_dir
"""

def make_ngrams(sentence, order):
  return collections.Counter(list(zip(*[sentence[i:] for i in range(order)])))

def ngram_overlap(summary, order):
    if order == 'sen':
        sum_ng = collections.Counter(tokenize.sent_tokenize(summary))
    else:
        sum_ng = make_ngrams(tokenize.word_tokenize(summary), order)
    total_ngrams = float(sum(sum_ng.values()))
    unique_ngrams = float(len(sum_ng.keys()))
    if total_ngrams == 0:
        return 0.0

    return (total_ngrams -unique_ngrams)/total_ngrams

def main(folder, stop_after):
    # Load summaries
    folder=os.path.join(folder, '*')
    sum_files = glob.glob(folder)
    sum_files = {int(os.path.basename(f)): f for f in sum_files}

    # Calculate novel ngrams and sentences for each summary
    ng_overlaps = {i:0 for i in list(range(1, 5)) + ['sen']}
    # ng_overlaps = {i:0 for i in list(range(1, 3))}
    files_done = 0
    for i in xrange(len(sum_files)):
        with open(sum_files[i], 'r') as f:
            summ = f.readlines()[-1].strip().split(' ', 1)[1].decode('utf-8')
            files_done += 1

        for o in ng_overlaps.keys():
            # TEST should be zero
            #no = ngram_overlap('The boy walks with his dog to the park', o)
            # TEST should be low, one bi-gram overlap, two single gram overlap
            #no = ngram_overlap('The boy walks with his dog his dog to the park', o)
            # TEST should be high, the first 'the' is non repetitve, all the others do overlap
            # no = ngram_overlap('the the the the the', o)

            no = ngram_overlap(summ, o)
            ng_overlaps[o] += no
        if files_done >= stop_after:
            break

    # Print results  
    print('Processed {} files'.format(files_done))
    for o, v in ng_overlaps.items():
        mean_o = ng_overlaps[o] / files_done
        print('%-duplicates {}-grams:\t{:.4f}'.format(o, mean_o*100))

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print('first arg should be result directory')
        exit(1)

    folder = sys.argv[1]
    stop_after = int(sys.argv[2]) if len(sys.argv) == 3 else 1e9
    main(folder, stop_after)
