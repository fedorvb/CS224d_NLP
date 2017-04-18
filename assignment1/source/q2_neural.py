import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive

def forward_backward_prop(data, labels, params, dimensions):
    """ 
    Forward and backward propagation for a two-layer sigmoidal network 
    
    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### YOUR CODE HERE: forward propagation
    h = sigmoid(np.dot(data, W1) + b1) # data=(N, Dx), W1=(Dx, H) => resulted dimension (N,H)    
    a1 = np.dot(h, W2) + b2    
    y_hat = softmax(a1) # h=(N, H), W2=(H, Dy) => resulted dimension (N,Dy)
    # cross entropy cost
    cost = -np.sum(labels*np.log(y_hat)) # dimension Dy
    ### END YOUR CODE
    
    ### YOUR CODE HERE: backward propagation
    grady = y_hat - labels # dimension (N, Dy)
    gradW2 = h.T.dot(grady) # h.T=(H,N), grady= (N, Dy) => resulted dimension (H, Dy) 
    gradb2 = np.sum(grady, axis=0) # colunm-wise sum, dimension Dy
    gradh = grady.dot(W2.T) # grady=(N, Dy), W2.T=(Dy, H) => resulted dimension (N, H)
    gradA1 = gradh * h * (1 - h) # element wise multiplication => resulted dimension (N, H)
    gradW1 = data.T.dot(gradA1) # data.T=(Dx, N), gradA1=(N,H) => resulted dimension (Dx,H)
    gradb1 = np.sum(gradA1, axis=0) # resulted dimension (H)
    ### END YOUR CODE
    
    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(), 
        gradW2.flatten(), gradb2.flatten()))
    
    return cost, grad

def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using 
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i,random.randint(0,dimensions[2]-1)] = 1
    
    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params: forward_backward_prop(data, labels, params,
        dimensions), params)

def your_sanity_checks(): 
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py 
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()