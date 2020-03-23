"""
Project: n-grams classify 
Class: CSC 439
Instructor: Bethard
Author: Lauren Olson
Description: This program has one function and three classes.

The function "read_smsspam" reads in a spam file and creates an array of tuples
where the first part of the tuple is a string that is the label of the sample and 
the second part is a string that is the sample feature.

The class "TextToFeatures" uses the "CountVectorizer" class to read in these 
features and create vectors 

The class "TextToLabels" uses the "LabelEncoder" class to read in the labels
and determine a vocabulary and create vectors associated with those labels

The class "Classifier" uses the "LogisticRegression" class as a classifier
which can both train with data and then make predictions

"""

from typing import Iterator, Iterable, Tuple, Text, Union

import numpy as np
from scipy.sparse import spmatrix
from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

NDArray = Union[np.ndarray, spmatrix]


def read_smsspam(smsspam_path: str) -> Iterator[Tuple[Text, Text]]:
    """Generates (label, text) tuples from the lines in an SMSSpam file.

    SMSSpam files contain one message per line. Each line is composed of a label
    (ham or spam), a tab character, and the text of the SMS. Here are some
    examples:

      spam	85233 FREE>Ringtone!Reply REAL
      ham	I can take you at like noon
      ham	Where is it. Is there any opening for mca.

    :param smsspam_path: The path of an SMSSpam file, formatted as above.
    :return: An iterator over (label, text) tuples.
    """
    #read file into an array 
    with open(smsspam_path) as f:
        file_array = f.readlines()
    array_of_tuples = create_tuples(file_array)
    
    c = Counter(array_of_tuples)

    return list(c.elements())[0:-2] #use yield keyword to generate iterators instead

def create_tuples(file_array):
    count = 0
    for line in file_array:
        line = line.split(None, 1)
        if(len(line) < 2):
                continue
        file_array[count] = (line[0], line[1])
        count += 1
    return file_array

class TextToFeatures:
    def __init__(self, texts: Iterable[Text]):
        """Initializes an object for converting texts to features.

        During initialization, the provided training texts are analyzed to
        determine the vocabulary, i.e., all feature values that the converter
        will support. Each such feature value will be associated with a unique
        integer index that may later be accessed via the .index() method.

        It is up to the implementer exactly what features to produce from a
        text, but the features will always include some single words and some
        multi-word expressions (e.g., "need" and "to you").

        :param texts: The training texts.
        """
        self.texts = texts
        self.vectorizer = CountVectorizer(binary = True, ngram_range=(1, 2),token_pattern=r'\b\w+\b', min_df=1) 
        #the vectorizer uses binary labels and 2 grams in order to optimize feature identification 
        analyze = self.vectorizer.build_analyzer()
        for text in texts:
            analyze(text)
        self.vectorizer.fit(texts)

                    

    def index(self, feature: Text):
        """Returns the index in the vocabulary of the given feature value.

        :param feature: A feature
        :return: The unique integer index associated with the feature.
        """
        count = 0
        for feature_name in self.vectorizer.get_feature_names():
            if(feature == feature_name):
                return count
            count += 1

    def __call__(self, texts: Iterable[Text]) -> NDArray:
        """Creates a feature matrix from a sequence of texts.

        Each row of the matrix corresponds to one of the input texts. The value
        at index j of row i is the value in the ith text of the feature
        associated with the unique integer j.

        It is up to the implementer what the value of a feature that is present
        in a text should be, though a common choice is 1. Features that are
        absent from a text will have the value 0.

        :param texts: A sequence of texts.
        :return: A matrix, with one row of feature values for each text.
        """
        return self.vectorizer.transform(texts) 
    


class TextToLabels:
    def __init__(self, labels: Iterable[Text]):
        """Initializes an object for converting texts to labels.

        During initialization, the provided training labels are analyzed to
        determine the vocabulary, i.e., all labels that the converter will
        support. Each such label will be associated with a unique integer index
        that may later be accessed via the .index() method.

        :param labels: The training labels.
        """
        self.le = LabelEncoder()
        self.le.fit(labels)

    def index(self, label: Text) -> int:
        """Returns the index in the vocabulary of the given label.

        :param label: A label
        :return: The unique integer index associated with the label.
        """
        count = 0
        for l in self.le.classes_:
            if(l == label):
                return count
            count += 1

    def __call__(self, labels: Iterable[Text]) -> NDArray:
        """Creates a label vector from a sequence of labels.

        Each entry in the vector corresponds to one of the input labels. The
        value at index j is the unique integer associated with the jth label.

        :param labels: A sequence of labels.
        :return: A vector, with one entry for each label.
        """
        return self.le.fit_transform(labels)
        


class Classifier:
    def __init__(self):
        """Initalizes a logistic regression classifier.
        """
        self.clf = LogisticRegression(solver='liblinear', multi_class='ovr', dual=True, max_iter=1000) 
        #the classifer uses liblinear, ovr and dual in order to maximize the classifier's ability 
        #to do binary classification
        
    def train(self, features: NDArray, labels: NDArray) -> None:
        """Trains the classifier using the given training examples.

        :param features: A feature matrix, where each row represents a text.
        Such matrices will typically be generated via TextToFeatures.
        :param labels: A label vector, where each entry represents a label.
        Such vectors will typically be generated via TextToLabels.
        """
        #matrix and target vector
        self.clf.fit(features, labels)

    def predict(self, features: NDArray) -> NDArray:
        """Makes predictions for each of the given examples.

        :param features: A feature matrix, where each row represents a text.
        Such matrices will typically be generated via TextToFeatures.
        :return: A prediction vector, where each entry represents a label.
        """

        return self.clf.predict(features)
