import numpy as np
import random

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid, sigmoid_grad

def normalizeRows(x):
    """ Row normalization function """
    # Implement a function that normalizes each row of a matrix to have unit length
    
    ### YOUR CODE HERE
    x = x / np.sqrt(np.sum(np.square(x), axis=1, keepdims=True))
    ### END YOUR CODE
    
    return x

def test_normalize_rows():
    print "Testing normalizeRows..."
    x = normalizeRows(np.array([[3.0,4.0],[1, 2]])) 
    # the result should be [[0.6, 0.8], [0.4472, 0.8944]]
    print x
    assert (x.all() == np.array([[0.6, 0.8], [0.4472, 0.8944]]).all())
    print ""

def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models """
    
    # Implement the cost and gradients for one predicted word vector  
    # and one target word vector as a building block for word2vec     
    # models, assuming the softmax prediction function and cross      
    # entropy loss.                                                   
    
    # Inputs:                                                         
    # - predicted: numpy ndarray, predicted word vector (\hat{v} in 
    #   the written component or \hat{r} in an earlier version)
    # - target: integer, the index of the target word               
    # - outputVectors: "output" vectors (as rows) for all tokens     
    # - dataset: needed for negative sampling, unused here.         
    
    # Outputs:                                                        
    # - cost: cross entropy cost for the softmax word prediction    
    # - gradPred: the gradient with respect to the predicted word   
    #        vector                                                
    # - grad: the gradient with respect to all the other word        
    #        vectors                                               
    
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!                                                  
    
    ### YOUR CODE HERE
    # Size V: vocabulary size
    # Size N: Vector Length
    # cost = -ylog(y^)  1xV
    #
    # ^y = p(o | c) = exp(uo.dot(vc)) / SUM_w=1_W{exp(uw.dot(vc)}  # softmax
    # uw denotes the w-th word and uw (w = 1; W) are the output word vectors 
    # for all words in the vocabulary.
    # where "uo" is the output word vector, and "vc" is the context word vector   
    # normalized by sum of all dot products of other word vectors with the context word vector
    #
    # y^ = softmax(vc*outputVector) 1xN * VxN = 1*V
    # after having a 1xV vector, take the target out
    # same as y^ multiple a one hot vector
    V, D = outputVectors.shape
    y = softmax(np.dot(predicted, outputVectors.T))
    # cost: cross entropy cost for the softmax word prediction   
    cost = -np.log(y[0, target])
    
    # grad CE and softmax will end up a form: y^ - label
    # dimension 1xV    
    y[0, target] = y[0, target] - 1
    #print "y.shape", y.shape
    #print "y", y
    
    # gradPred: the gradient with respect to the predicted word vector
    # dimension y=(1,V), outputVectors=(V, D) => resulted dimension (1,D)
    gradPred = np.dot(y, outputVectors) 
    
    # - grad: the gradient with respect to all the other word vectors     
    # dimension y.T=(V,1), predicted=(1, D) => resulted dimension (V,D)
    grad = np.dot(y.T, predicted.reshape(1, D))
    ### END YOUR CODE
    
    return cost, gradPred, grad

def negSamplingCostAndGradient(predicted, target, outputVectors, dataset, 
    K=10):
    """ Negative sampling cost function for word2vec models """

    # Implement the cost and gradients for one predicted word vector  
    # and one target word vector as a building block for word2vec     
    # models, using the negative sampling technique. K is the sample  
    # size. You might want to use dataset.sampleTokenIdx() to sample  
    # a random word index. 
    # 
    # Note: See test_word2vec below for dataset's initialization.
    #                                       
    # Input/Output Specifications: same as softmaxCostAndGradient     
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!
    
    ### YOUR CODE HERE
    # outputVectors=(V, D), predicted=(D)        
    scores = sigmoid(np.dot(outputVectors, predicted))
    sampled_tokens = [dataset.sampleTokenIdx() for i in range(K)]
    
    # negative sampling cost
    # cost = - log(sigmoid(predicted * output[target]))
    #         - sum(log(sigmoid(predicted * - output[sample])))    
    # dimension of cost is (V) 
    
    # scores[sampled_tokens] == sigmoid(predicted * - output[sample]) == 
    # == 1 - sigmoid(predicted * output[sample])
    # basic property of sigmoid(-X) = 1 - sigmoid(X)
    cost = - np.log(scores[target]) - np.sum(np.log(1 - scores[sampled_tokens]))
    # reminder: scores[sampled_tokens] = sigmoid(np.dot(-samples, predicted))

    # gradPred: the gradient with respect to the predicted word vector
    # derivative of "- np.log(scores[target])" = -1/(scores[target]) * 
    # * (scores[target]) * (1-(scores[target])) * outputVectors[target]
    gradPred = (scores[target] - 1) * outputVectors[target] \
               + np.sum(outputVectors[sampled_tokens].T * scores[sampled_tokens], axis=1)

    # - grad: the gradient with respect to all the other word vectors 
    grad = np.zeros_like(outputVectors)
    grad[target] = (scores[target]-1) * predicted
    for token in sampled_tokens:
        grad[token] += scores[token] * predicted

    ### END YOUR CODE
    return cost, gradPred, grad
   
def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors, 
    dataset, word2vecCostAndGradient = softmaxCostAndGradient):
    """ Skip-gram model in word2vec """

    # Implement the skip-gram model in this function.

    # Inputs:                                                         
    # - currrentWord: a string of the current center word           
    # - C: integer, context size                                    
    # - contextWords: list of no more than 2*C strings, the context words                                               
    # - tokens: a dictionary that maps words to their indices in    
    #      the word vector list                                
    # - inputVectors: "input" word vectors (as rows) for all tokens           
    # - outputVectors: "output" word vectors (as rows) for all tokens         
    # - word2vecCostAndGradient: the cost and gradient function for 
    #      a prediction vector given the target word vectors,  
    #      could be one of the two cost functions you          
    #      implemented above

    # Outputs:                                                        
    # - cost: the cost function value for the skip-gram model       
    # - grad: the gradient with respect to the word vectors         
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!

    ### YOUR CODE HERE
    cost = 0
    currentVector = inputVectors[tokens[currentWord]]
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)
    for context in contextWords:
        c, gPred, g = word2vecCostAndGradient(currentVector, tokens[context], outputVectors, dataset)
        cost += c
        gradIn[tokens[currentWord]] += gPred.flatten()
        gradOut += g
    
    ### END YOUR CODE
    
    return cost, gradIn, gradOut

def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors, 
    dataset, word2vecCostAndGradient = softmaxCostAndGradient):
    """ CBOW model in word2vec """

    # Implement the continuous bag-of-words model in this function.            
    # Input/Output specifications: same as the skip-gram model        
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!

    #################################################################
    # IMPLEMENTING CBOW IS EXTRA CREDIT, DERIVATIONS IN THE WRIITEN #
    # ASSIGNMENT ARE NOT!                                           #  
    #################################################################
    
    cost = 0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE   

    # For (a simpler variant of) CBOW, we sum up the input word vectors in the context
    # v^ = SUM_j=-m,m {vc_j} - this is a sum of the context vectors around the current word(2*m vectors)
    total_cost = 0
    total_grad_in = np.zeros_like(inputVectors)
    total_grad_out = np.zeros_like(outputVectors)
    input_vector = np.sum([inputVectors[tokens[context_word]]
                           for context_word in contextWords], axis=0)  # v_c
    target = tokens[currentWord] # y
    # in simplified CBOW model we predict "sum of vectors the context words", 
    # and we use the center word to predict context words around it    
    cost, grad_predicted, grad_out = word2vecCostAndGradient(input_vector, target, outputVectors,
                                                             dataset)
    total_cost += cost
    for context_word in contextWords:
        #print "grad_predicted.shape", grad_predicted.shape
        total_grad_in[tokens[context_word]] += grad_predicted.flatten()
    total_grad_out += grad_out
    
    ### END YOUR CODE
    
    return total_cost, total_grad_in, total_grad_out

#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C, word2vecCostAndGradient = softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N/2,:]
    outputVectors = wordVectors[N/2:,:]
    for i in xrange(batchsize):
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1)
        
        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1
        
        c, gin, gout = word2vecModel(centerword, C1, context, tokens, inputVectors, outputVectors, dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N/2, :] += gin / batchsize / denom
        grad[N/2:, :] += gout / batchsize / denom
        
    return cost, grad

def test_word2vec():
    # Interface to the dataset for negative sampling
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], [tokens[random.randint(0,4)] \
           for i in xrange(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    print "==== Gradient check for skip-gram ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5), dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)
    print "\n==== Gradient check for CBOW      ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5), dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)

    print "\n=== Results ==="
    print skipgram("c", 3, ["a", "b", "e", "d", "b", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print skipgram("c", 1, ["a", "b"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset, negSamplingCostAndGradient)
    print cbow("a", 2, ["a", "b", "c", "a"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print cbow("a", 2, ["a", "b", "a", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset, negSamplingCostAndGradient)

if __name__ == "__main__":
    test_normalize_rows()
    test_word2vec()