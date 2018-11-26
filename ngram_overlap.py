from nltk import tokenize

def make_ngrams(sentence, order):
  return zip(*[sentence[i:] for i in range(order)])

def ngram_overlap(summary, doc, order):
    sum_ng = set()
    doc_ng = set()

    sum_ng = set(make_ngrams(tokenize.word_tokenize(summary), order))
    doc_ng = set(make_ngrams(tokenize.word_tokenize(doc), order))
    overlap = sum_ng.intersection(doc_ng)
    print  "{:.2f}% novel {}-grams".format((1 - len(overlap) / float(len(sum_ng)))*100, order)

def sentence_overlap(summary, doc):
    sum_s = set(tokenize.sent_tokenize(summary))
    doc_s = set(tokenize.sent_tokenize(doc))
    overlap = sum_s.intersection(doc_s)
    print  "{:.2f}% novel sentences".format((1 - len(overlap) / float(len(sum_s)))*100)

if __name__ == "__main__":
    doc = "senate republicans are in the final stages of producing a sweeping rewrite of the nation's laws after weeks of highly secretive deliberations and lingering frustration among the rank and file over how to fulfill the party's signature campaign promise of the past seven years. senate majority leader mitch mcconnell ( .) said tuesday that the gop leadership will produce a 'discussion draft' of the bill on thursday."
    summary = "senate gop leaders will present health bill this week, even as divisions flare."
    for order in range(1, 5):
        ngram_overlap(summary, doc, order)
    sentence_overlap(summary, doc)
