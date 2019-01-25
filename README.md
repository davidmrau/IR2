# Towards more abstractive text summarization with Pointer-Generator Networks
## Abstract
The Pointer-Generator architecture has shown to be a big improve- ment for abstractive summarization seq2seq models. However, the summaries produced by this model are largely extractive as over 30% of the generated sentences are copies from the source text. This work proposes a multihead attention mechanism, pointer dropout and two new loss functions to promote more abstractive summaries while maintaining similar ROUGE scores. Both the multihead at- tention and dropout do not improve N-gram novelty, however, the dropout acts as a regularizer which improves the ROUGE score. The new loss function achieve significantly higher novel N-grams and sentences, at the cost of a slightly lower ROUGE score.

The full report can be found [here]().

The code is based on a Pointer-Generator implementation which can be found [here](https://github.com/lipiji/neural-summ-cnndm-pytorch).
