
"""This file shows a couple of implementations of the perceptron learning
algorithm. It is based on the code from Lecture 3, but using the slightly
more compact perceptron formulation that we saw in Lecture 6.

There are two versions: Perceptron, which uses normal NumPy vectors and
matrices, and SparsePerceptron, which uses sparse vectors and matrices.
The latter may be faster when we have high-dimensional feature representations
with a lot of zeros, such as when we are using a "bag of words" representation
of documents.
"""

import numpy as np
from sklearn.base import BaseEstimator

class LinearClassifier(BaseEstimator):
    """
    General class for binary linear classifiers. Implements the predict
    function, which is the same for all binary linear classifiers. There are
    also two utility functions.
    """

    def decision_function(self, X):
        """
        Computes the decision function for the inputs X. The inputs are assumed to be
        stored in a matrix, where each row contains the features for one
        instance.
        """
        return X.dot(self.w)

    def predict(self, X):
        """
        Predicts the outputs for the inputs X. The inputs are assumed to be
        stored in a matrix, where each row contains the features for one
        instance.
        """

        # First compute the output scores
        scores = self.decision_function(X)

        # Select the positive or negative class label, depending on whether
        # the score was positive or negative.
        out = np.select([scores >= 0.0, scores < 0.0],
                        [self.positive_class,
                         self.negative_class])
        return out

    def find_classes(self, Y):
        """
        Finds the set of output classes in the output part Y of the training set.
        If there are exactly two classes, one of them is associated to positive
        classifier scores, the other one to negative scores. If the number of
        classes is not 2, an error is raised.
        """
        classes = sorted(set(Y))
        if len(classes) != 2:
            raise Exception("this does not seem to be a 2-class problem")
        self.positive_class = classes[1]
        self.negative_class = classes[0]

    def encode_outputs(self, Y):
        """
        A helper function that converts all outputs to +1 or -1.
        """
        return np.array([1 if y == self.positive_class else -1 for y in Y])


class Perceptron(LinearClassifier):
    """
    A straightforward implementation of the perceptron learning algorithm.
    """

    def __init__(self, n_iter=20):
        """
        The constructor can optionally take a parameter n_iter specifying how
        many times we want to iterate through the training set.
        """
        self.n_iter = n_iter

    def fit(self, X, Y):
        """
        Train a linear classifier using the perceptron learning algorithm.
        """

        # First determine which output class will be associated with positive
        # and negative scores, respectively.
        self.find_classes(Y)

        # Convert all outputs to +1 (for the positive class) or -1 (negative).
        Ye = self.encode_outputs(Y)

        # If necessary, convert the sparse matrix returned by a vectorizer
        # into a normal NumPy matrix.
        if not isinstance(X, np.ndarray):
            X = X.toarray()

        # Initialize the weight vector to all zeros.
        n_features = X.shape[1]
        self.w = np.zeros(n_features)

        # Perceptron algorithm:
        for i in range(self.n_iter):
            for x, y in zip(X, Ye):

                # Compute the output score for this instance.
                score = x.dot(self.w)

                # If there was an error, update the weights.
                if y*score <= 0:
                    self.w += y*x


##### The following part is for the optional task.

### Sparse and dense vectors don't collaborate very well in NumPy/SciPy.
### Here are two utility functions that help us carry out some vector
### operations that we'll need.

def add_sparse_to_dense(x, w, factor):
    """
    Adds a sparse vector x, scaled by some factor, to a dense vector.
    This can be seen as the equivalent of w += factor * x when x is a dense
    vector.
    """
    w[x.indices] += factor * x.data

def sparse_dense_dot(x, w):
    """
    Computes the dot product between a sparse vector x and a dense vector w.
    """
    return np.dot(w[x.indices], x.data)


class SparsePerceptron(LinearClassifier):
    """
    A straightforward implementation of the perceptron learning algorithm,
    assuming that the input feature matrix X is sparse.
    """

    def __init__(self, n_iter=20):
        """
        The constructor can optionally take a parameter n_iter specifying how
        many times we want to iterate through the training set.
        """
        self.n_iter = n_iter

    def fit(self, X, Y):
        """
        Train a linear classifier using the perceptron learning algorithm.

        Note that this will only work if X is a sparse matrix, such as the
        output of a scikit-learn vectorizer.
        """
        self.find_classes(Y)

        # First determine which output class will be associated with positive
        # and negative scores, respectively.
        Ye = self.encode_outputs(Y)

        # Initialize the weight vector to all zeros.
        self.w = np.zeros(X.shape[1])

        # Iteration through sparse matrices can be a bit slow, so we first
        # prepare this list to speed up iteration.
        XY = list(zip(X, Ye))

        for i in range(self.n_iter):
            for x, y in XY:

                # Compute the output score for this instance.
                # (This corresponds to score = x.dot(self.w) above.)
                score = sparse_dense_dot(x, self.w)

                # If there was an error, update the weights.
                if y*score <= 0:
                    # (This corresponds to self.w += y*x above.)
                    add_sparse_to_dense(x, self.w, y)


class LinearSVC(LinearClassifier):
    """
    An implementation of the PEGASOS algortigm for linear support vector classifier
    """

    def __init__(self, n_iter=10, weight=1, normalize=False, verbose=False):
        """
        The constructor can optionally take a parameter n_iter specifying how
        many times we want to iterate through the training set. Floats allowed. E.g. 1.5 or 0.5 times
        Weight the value of the size of w in the object function to minimize
        Normalize W on each iteration to stay within the ball of radius 1/sqrt(weight)
        """
        self.n_iter = n_iter
        self.weight = weight
        self.normalize = normalize
        self.verbose = verbose

    def fit(self, X, Y):
        """
        Train a linear classifier using the PEGASOS learning algorithm.
        """

        # First determine which output class will be associated with positive
        # and negative scores, respectively.
        self.find_classes(Y)

        # Convert all outputs to +1 (for the positive class) or -1 (negative).
        Ye = self.encode_outputs(Y)

        # If necessary, convert the sparse matrix returned by a vectorizer
        # into a normal NumPy matrix.
        if not isinstance(X, np.ndarray):
            X = X.toarray()

        # Initialize the weight vector to all zeros.
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)

        if self.verbose:
            print("Epoch | Objective | Loss\n-------------------------")

        # Pegasos algorithm:
        for e in range(self.n_iter):
            for t in range(1, n_samples + 1):
                #Draw random sample from training data
                i = np.random.randint(0, n_samples)
                y, x = Ye[i], X[i]

                #calculate step-size
                eta = 1/(self.weight*(t + e*n_samples))
                #predict for this sample
                score = x.dot(self.w)
                #adjust w
                self.w *= 1-eta*self.weight
                #If we predict wrongly or with to small margin
                if y*score < 1:
                    #adjust w
                    self.w += eta*y*x

                #optional
                if self.normalize:
                    #limit w to ball with radius 1/sqrt(weight)
                    self.w *= min(1, (1/self.weight**0.5)/np.linalg.norm(self.w))
            
            if self.verbose:
                loss = np.sum(np.maximum(0, 1 - Ye*X.dot(self.w)))
                objective = self.weight/2*np.linalg.norm(self.w)**2 + loss/n_samples
                print(f"  {e}  |  {objective:.2f}  |  {loss:.2f}")                    

class LogisticRegression(LinearClassifier):
    """
    Logistic Regression classifier
    """

    def __init__(self, n_iter=10, weight=1, normalize=False, verbose=False):
        """
        The constructor can optionally take a parameter n_iter specifying how
        many times we want to iterate through the training set. Floats allowed. E.g. 1.5 or 0.5 times
        Weight the value of the size of w in the object function to minimize
        Normalize W on each iteration to stay within the ball of radius 1/sqrt(weight)
        """
        self.n_iter = n_iter
        self.weight = weight
        self.normalize = normalize
        self.verbose = verbose



    def fit(self, X, Y):
        """
        Pegasos logistic regression
        """
        self.find_classes(Y)
        Ye = self.encode_outputs(Y)
        if not isinstance(X, np.ndarray):
            X = X.toarray()

        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)

        if self.verbose:
            print("Epoch | Objective | Loss\n-------------------------")

        for e in range(self.n_iter):
            for t in range(1, n_samples + 1):
                #Draw random sample from training data
                i = np.random.randint(0, n_samples)
                y, x = Ye[i], X[i]

                #calculate step-size
                eta = 1/(self.weight*(t + e*n_samples))
                #predict for this sample
                score = x.dot(self.w)
                #adjust w
                self.w = (1-eta*self.weight)*self.w + eta*y/(1+np.exp(y*score))*x

                #optional
                if self.normalize:
                    #limit w to ball with radius 1/sqrt(weight)
                    self.w *= min(1, (1/self.weight**0.5)/np.linalg.norm(self.w))

            if self.verbose:
                loss = np.sum(np.log(1+np.exp(-Ye * X.dot(self.w))))
                objective = self.weight/2*np.linalg.norm(self.w)**2 + loss/n_samples
                print(f"  {e}  |  {objective:.2f}  |  {loss:.2f}")