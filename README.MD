<div id="header">[![](http://nlp.stanford.edu/sentiment/images/nlp-logo.gif) ](http://nlp.stanford.edu/) [

# CS224d: Deep Learning for Natural Language Processing

](index.html)</div>

<div class="sechighlight">

<div class="container sec">

## Schedule and Syllabus

Unless otherwise specified the course lectures and meeting times are:

Tuesday, Thursday 3:00-4:20
Location: Gates B1</div>

</div>

<div class="container sec">

| Event | Date | Description | Course Materials |
| Lecture | Mar 29 | Intro to NLP and Deep Learning | Suggested Readings:

1.  [[Linear Algebra Review](http://cs229.stanford.edu/section/cs229-linalg.pdf)]
2.  [[Probability Review](http://cs229.stanford.edu/section/cs229-prob.pdf)]
3.  [[Convex Optimization Review](http://cs229.stanford.edu/section/cs229-cvxopt.pdf)]
4.  [[More Optimization (SGD) Review](http://cs231n.github.io/optimization-1/)]
5.  [[From Frequency to Meaning: Vector Space Models of Semantics](http://www.jair.org/media/2934/live-2934-4846-jair.pdf)]

[[Lecture Notes 1](lecture_notes/notes1.pdf)]
[[python tutorial](http://cs231n.github.io/python-numpy-tutorial/)] [[slides](lectures/CS224d-Lecture1.pdf)] |
| Lecture | Mar 31 | Simple Word Vector representations: word2vec, GloVe | Suggested Readings:

1.  [[Distributed Representations of Words and Phrases and their Compositionality](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)]
2.  [[Efficient Estimation of Word Representations in Vector Space](http://arxiv.org/pdf/1301.3781.pdf)]

<div>[[slides](lectures/CS224d-Lecture2.pdf)]</div>

 |
| A1 released | Apr 4 | **Pset #1 released** | [[Pset 1](assignment1/index.html)] [[Pset 1 Solutions](assignment1/assignment1_soln)] [[Pset 1 Solutions Code](assignment1/assignment1_sol.zip)] |
| Lecture | Apr 5 | Advanced word vector representations: language models, softmax, single layer networks | Suggested Readings:

1.  [[GloVe: Global Vectors for Word Representation](http://nlp.stanford.edu/pubs/glove.pdf)]
2.  [[Improving Word Representations via Global Context and Multiple Word Prototypes](http://www.aclweb.org/anthology/P12-1092)]

<div>
[[Lecture Notes 2](lecture_notes/notes2.pdf)]
[[slides](lectures/CS224d-Lecture3.pdf)]</div>

 |
| Lecture | Apr 7 | Neural Networks and backpropagation -- for named entity recognition | Suggested Readings:

1.  [[UFLDL tutorial](http://ufldl.stanford.edu/wiki/index.php/Backpropagation_Algorithm)]
2.  [[Learning Representations by Backpropogating Errors](http://www.iro.umontreal.ca/~vincentp/ift3395/lectures/backprop_old.pdf)]

[[Lecture Notes 3](lecture_notes/notes3.pdf)]
[[slides](lectures/CS224d-Lecture4.pdf)] |
| Lecture | Apr 12 | Project Advice, Neural Networks and Back-Prop (in full gory detail) | Suggested Readings:

1.  [[Natural Language Processing (almost) from Scratch](http://arxiv.org/pdf/1103.0398v1.pdf)]
2.  [[A Neural Network for Factoid Question Answering over Paragraphs](https://cs.umd.edu/~miyyer/pubs/2014_qb_rnn.pdf)]
3.  [[Grounded Compositional Semantics for Finding and Describing Images with Sentences](http://nlp.stanford.edu/~socherr/SocherKarpathyLeManningNg_TACL2013.pdf)]
4.  [[Deep Visual-Semantic Alignments for Generating Image Descriptions](http://cs.stanford.edu/people/karpathy/deepimagesent/devisagen.pdf)]
5.  [[Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank](http://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf)]

<div>
[[slides](lectures/CS224d-Lecture5.pdf)]</div>

 |
| Lecture | Apr 14 | Practical tips: gradient checks, overfitting, regularization, activation functions, details | Suggested Readings:

1.  [[Practical recommendations for gradient-based training of deep architectures](http://arxiv.org/abs/1206.5533)]
2.  [[UFLDL page on gradient checking](http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization)]

[[slides](lectures/CS224d-Lecture6.pdf)] |
| A1 Due | Apr 19 | **Pset #1 due** |
| Lecture | Apr 19 | Introduction to Tensorflow | Suggested Readings:

1.  [[TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems](http://download.tensorflow.org/paper/whitepaper2015.pdf)]

[[slides](lectures/CS224d-Lecture7.pdf)] [[AWS Tutorial](supplementary/aws-tutorial-2.pdf)] [[AWS Tutorial Supplementary](lectures/CS224D-Lecture7-2.pdf)] [[AWS Tutorial Video](https://youtu.be/zdnMXKHP-m4)]
 |
| A2 released | Apr 20 | **Pset #2 released** | [[Pset 2](assignment2/index.html)][[Pset 2 Solutions](assignment2/assignment2_sol.pdf)] [[Pset 2 Solutions Code](assignment2/assignment2_dev.zip)] |
| Lecture | Apr 21 | Recurrent neural networks -- for language modeling and other tasks | Suggested Readings:

1.  [[Recurrent neural network based language model](http://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf)]
2.  [[Extensions of recurrent neural network language model](http://www.fit.vutbr.cz/research/groups/speech/publi/2011/mikolov_icassp2011_5528.pdf)]
3.  [[Opinion Mining with Deep Recurrent Neural Networks](http://www.cs.cornell.edu/~oirsoy/drnt.htm)]

<div>
[[slides](lectures/CS224d-Lecture8.pdf)] [[minimal net example (karpathy)](http://cs231n.github.io/neural-networks-case-study/)] [[vanishing grad example](notebooks/vanishing_grad_example.html)] [[vanishing grad notebook](notebooks/vanishing_grad_example.ipynb)]
[[Lecture Notes 4](lecture_notes/notes4.pdf)]
</div>

 |
| Lecture | Apr 26 | GRUs and LSTMs -- for machine translation | Suggested Readings:

1.  [[Long Short-Term Memory](http://web.eecs.utk.edu/~itamar/courses/ECE-692/Bobby_paper1.pdf)]
2.  [[Gated Feedback Recurrent Neural Networks](http://arxiv.org/pdf/1502.02367v3.pdf)]
3.  [[Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](http://arxiv.org/pdf/1412.3555v1.pdf)]

[[slides](lectures/CS224d-Lecture9.pdf)] |
| Proposal due | **Apr 28** | **Course Project Proposal due** | [[proposal description](project.html#proposal)] |
| Lecture | Apr 28 | Recursive neural networks -- for parsing | Suggested Readings:

1.  [[Parsing with Compositional Vector Grammars](http://nlp.stanford.edu/pubs/SocherBauerManningNg_ACL2013.pdf)]
2.  [[Subgradient Methods for Structured Prediction](http://repository.cmu.edu/cgi/viewcontent.cgi?article=1054&context=robotics)]
3.  [[Parsing Natural Scenes and Natural Language with Recursive Neural Networks](http://www-nlp.stanford.edu/pubs/SocherLinNgManning_ICML2011.pdf)]

[[Lecture Notes 5](lecture_notes/LectureNotes5.pdf)]
[[slides](lectures/CS224d-Lecture10.pdf)] |
| Lecture | May 3 | Recursive neural networks -- for different tasks (e.g. sentiment analysis) | Suggested Readings:

1.  [[Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank](http://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf)]
2.  [[Dynamic Pooling and Unfolding Recursive Autoencoders for Paraphrase Detection](http://papers.nips.cc/paper/4204-dynamic-pooling-and-unfolding-recursive-autoencoders-for-paraphrase-detection.pdf)]
3.  [[Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks](http://arxiv.org/pdf/1503.00075v2.pdf)]

[[slides](lectures/CS224d-Lecture11.pdf)] |
| A2 Due | **May 5** | **Pset #2 Due date** |
| Lecture | May 5 | Review Session for Midterm | 

Suggested Readings: N/A

[[slides](lectures/CS224d-Lecture12.pdf)] |
| Midterm | May 10 | In-class midterm | [[midterm solutions](midterm/midterm_solutions.pdf)] |
| A3 released | May 12 | **Pset #3 released** | [[Pset 3](assignment3/index.html)] [[Pset 3 Solutions](assignment3/pset3_soln.pdf)] [[Pset 3 Solutions Code](assignment3/pset3_code.zip)] |
| Lecture | May 12 | Convolutional neural networks -- for sentence classification | Suggested Readings:

1.  [[A Convolutional Neural Network for Modelling Sentences](http://nal.co/papers/Kalchbrenner_DCNN_ACL14)]

[[slides](lectures/CS224d-Lecture13.pdf)] |
| Milestone | **May 15** | Course Project Milestone | [[milestone description](project.html#milestone)] |
| Lecture | May 17 | Guest Lecture with [Andrew Maas](http://ai.stanford.edu/~amaas/): Speech recognition | Suggested Readings:

1.  [ [Deep Neural Networks for Acoustic Modeling in Speech Recognition](papers/maas_paper.pdf)]

[[slides](lectures/CS224d-Lecture14.pdf)] |
| Lecture | May 19 | Guest Lecture with [Thang Luong](http://stanford.edu/~lmthang/): Machine Translation | Suggested Readings:

1.  [ [Achieving Open Vocabulary Neural Machine Translation with Hybrid Word-Character Models](papers/achieving.pdf)]
2.  [ [Addressing the Rare Word Problem in Neural Machine Translation](papers/addressing.pdf)]
3.  [ [Advances in natural language processing](papers/advances.pdf)]
4.  [ [Neural machine translation by jointly learning to align and translate](papers/neural_machine.pdf)]

[[slides](lectures/CS224d-Lecture15.pdf)] |
| A3 Due | May 21 | **Pset #3 Due date** |
| Lecture | May 24 | Guest Lecture with [Quoc Le](http://cs.stanford.edu/~quocle/): Seq2Seq and Large Scale DL | Suggested Readings:

1.  [ [Sequence to Sequence with Neural Networks](papers/seq2seq.pdf)]
2.  [ [Neural Machine Translation by Jointly Learning to Align and Translate](papers/nmt.pdf)]
3.  [ [A Neural Conversation Model](papers/ancm.pdf)]
4.  [ [Neural Programmer: Include Latent Programs with Gradient Descent](papers/npil.pdf)]

[[slides](lectures/CS224d-Lecture16.pdf)] |
| Lecture | May 26 | The future of Deep Learning for NLP: Dynamic Memory Networks | Suggested Readings:

1.  [[Ask me anthing: Dynamic Memory Networks for NLP](http://arxiv.org/abs/1506.07285)]

[[slides](lectures/CS224d-Lecture17.pdf)] |
| Poster Presentation | June 1 | Final project poster presentations: 2-5 pm, Gates patio |
| Final Project Due | **Jun 3** | Final course project due date | [project description] |

</div>