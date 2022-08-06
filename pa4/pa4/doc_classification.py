
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import LinearSVC as LSVC
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from aml_perceptron import Perceptron, SparsePerceptron, LinearSVC, LogisticRegression

# This function reads the corpus, returns a list of documents, and a list
# of their corresponding polarity labels. 
def read_data(corpus_file):
    X = []
    Y = []
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            _, y, _, x = line.split(maxsplit=3)
            X.append(x.strip())
            Y.append(y)
    return X, Y


if __name__ == '__main__':
    
    # Read all the documents.
    X, Y = read_data('data/all_sentiment_shuffled.txt')
    
    # Split into training and test parts.
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2,
                                                    random_state=0)

    # Set up the preprocessing steps and the classifier.
    pipeline = make_pipeline(
        TfidfVectorizer(),
        # SelectKBest(k=1000),
        Normalizer(),

        # NB that this is our Perceptron, not sklearn.linear_model.Perceptron
        Perceptron()  
    )

    pipelineSparse = make_pipeline(
        TfidfVectorizer(),
        # SelectKBest(k=1000),
        Normalizer(),

        # NB that this is our Perceptron, not sklearn.linear_model.Perceptron
        SparsePerceptron()  
    )

    pipeline2 = make_pipeline(
        TfidfVectorizer(),
        # SelectKBest(k=1000),
        Normalizer(),

        # NB that this is our LinearSVC, not sklearn.linear_model.LinearSVC
        LinearSVC(n_iter=10, weight=1/len(Xtrain), normalize=True, verbose=True)  
    )

    pipeline3 = make_pipeline(
        TfidfVectorizer(),
        # SelectKBest(k=1000),
        Normalizer(),

        # NB that this is our LogisticRegression, not sklearn.linear_model.LogisticRegression
        LogisticRegression(n_iter=10, weight=1/len(Xtrain), normalize=True, verbose=True)  
    )

    pipelineComp = make_pipeline(
        TfidfVectorizer(),
        # SelectKBest(k=1000),
        Normalizer(),

        #sklearn.linear_model.LogisticRegression
        LR()  
    )

    pipelineComp1 = make_pipeline(
        TfidfVectorizer(),
        # SelectKBest(k=1000),
        Normalizer(),

        #sklearn.linear_model.LinearSVC
        LSVC()  
    )

    # Train the classifier.
    t0 = time.time()
    pipeline.fit(Xtrain, Ytrain)
    t1 = time.time()
    print('Perceptron Training time: {:.2f} sec.'.format(t1-t0))

    # Evaluate on the test set.
    Yguess = pipeline.predict(Xtest)
    print('Perceptron Accuracy: {:.4f}.'.format(accuracy_score(Ytest, Yguess)))

    # Train the classifier.
    t0 = time.time()
    pipelineSparse.fit(Xtrain, Ytrain)
    t1 = time.time()
    print('PerceptronSparse Training time: {:.2f} sec.'.format(t1-t0))

    # Evaluate on the test set.
    Yguess = pipelineSparse.predict(Xtest)
    print('PerceptronSparse Accuracy: {:.4f}.'.format(accuracy_score(Ytest, Yguess)))

    # Train the classifier.
    t0 = time.time()
    pipeline2.fit(Xtrain, Ytrain)
    t1 = time.time()
    print('SVC Training time: {:.2f} sec.'.format(t1-t0))

    # Evaluate on the test set.
    Yguess = pipeline2.predict(Xtest)
    print('SVC Accuracy: {:.4f}.'.format(accuracy_score(Ytest, Yguess)))

    # Train the classifier.
    t0 = time.time()
    pipelineComp1.fit(Xtrain, Ytrain)
    t1 = time.time()
    print('scikit SVC Training time: {:.2f} sec.'.format(t1-t0))

    # Evaluate on the test set.
    Yguess = pipelineComp1.predict(Xtest)
    print('scikit SVC Accuracy: {:.4f}.'.format(accuracy_score(Ytest, Yguess)))

    # Train the classifier.
    t0 = time.time()
    pipeline3.fit(Xtrain, Ytrain)
    t1 = time.time()
    print('LogReg Training time: {:.2f} sec.'.format(t1-t0))

    # Evaluate on the test set.
    Yguess = pipeline3.predict(Xtest)
    print('LogReg Accuracy: {:.4f}.'.format(accuracy_score(Ytest, Yguess)))

    # Train the classifier.
    t0 = time.time()
    pipelineComp.fit(Xtrain, Ytrain)
    t1 = time.time()
    print('Scikit LR Training time: {:.2f} sec.'.format(t1-t0))

    # Evaluate on the test set.
    Yguess = pipelineComp.predict(Xtest)
    print('Scikit LR Accuracy: {:.4f}.'.format(accuracy_score(Ytest, Yguess)))

