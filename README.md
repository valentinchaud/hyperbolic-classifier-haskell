# hyperbolic-classifier


## Introduction
This repository is an attempt to implement a hyperbolic classifier for determining, for any pair of words, if the first
word is a hypernym (i.e has a broader meaning. For example, "animal" is a hypernym for the word "dog") of the second.

Hyperbolic classifier, in this specific scenario, solves a known problem in euclidian geometry : because the relations
of words are hierarchical, the nodes will grow exponentially by going deeper into the hierarchy while euclidian spaces
grow in a linear space (with the radius). For this reason, if we want to fit the hierarchy into a linear space, we will
need to make the nodes closer between them. This distance reduction will lead at some point to have similarities between
multiple nodes that doesn't represent similar concept. Because hyperbolic space grows exponentially, it avoids the
problem we meet in euclidian space.

## Files in the repo

This repo consisting of multiple files:

- src/Wordnet: An updated library for parsing the WordNet database.
- src/HyperbolicEmb.hs: an attempt to implement hyperbolic embeddings using the Poincarré ball, a model of hyperbolic geometry where points are in a unit ball that grow exponentially as we approach its boundaries. While it implements core functions like mobius operation, riemannian gradients, exponential/log mapping, I suspect that part of the library wouldn't be implemented correctly, so it has to be considerated with attention.
- app/HyperbolicTraining.hs : An attempt to train hyperbolic classifier that currently doesn't work, the loss going to NaN after only 1 or 2 batches.
- app/HypernymMLP.hs : An attempt to train a hypernym classifier using MLP, that currently work for the same reason.


## My recommendations for future work on that project

There are multiple things to verify if we want to make the project work in the future :

- First, the dataset generated to try to train the classifier may be wrong. It's in fact composed only of three elements :
  the　first word, the second word and if the first word is a hypernym of the second word. I think that having only these
  elements will make the model difficult to converge, as they have nearly no element to base for the extrapolation.
  However, it wouldn't explain itself, in my opinion, why we have NaN values so early in the training
- The hyperbolic embeddings could also have problems, but I can't find any specific part that wouldn't work, and it
  wouldn't explain why the MLP training doesn't work either.
- The hyperbolic training was coded at the end of the internship, and had multiples attempt to fix the training that
  ultimately didn't work. Recoding the training entirely could be a good idea.

Testing the MLP and hyperbolic embeddings on another dataset could determine if the problem is from the dataset or other
part of the code, so it would be the first thing to do before checking about the code itself.

