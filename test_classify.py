import numpy as np
import pytest
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from sklearn_porter import Porter

#classifiers
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

#vectorizers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import HashingVectorizer

import classify


@pytest.fixture(autouse=True)
def set_seeds():
    np.random.seed(42)


def test_read_smsspam():
    # keep a counter here (instead of enumerate) in case the iterator is empty
    count = 0
    for example in classify.read_smsspam("AGBIG_annotation.outt"):

        # make sure the right shape is returned
        assert len(example) == 2
        label, text = example

        # make sure the label is one of the expected two
        assert label in {"yes", "no"}

        count += 1
    assert count == 2998


def test_features():
    # get the texts from the training data
    examples = classify.read_smsspam("AGBIG_annotation.outt")
    texts = [text for _, text in examples]

    # create the feature extractor from the training texts
    to_features = classify.TextToFeatures(texts)

    # extract features for some made-up sentences
    features = to_features(["There are some things that I need to send to you.",
                            "Hello!"])

    # make sure there is one row of features for each sentence
    assert len(features.shape) == 2
    n_rows, n_cols = features.shape
    assert n_rows == 2

    # make sure there are nonzero values for some selected unigram and bigram
    # features in the first sentence
    indices = [to_features.index(f) for f in ["need", "to you"]]
    assert len(set(indices)) > 1
    row_indices, col_indices = features[:, indices].nonzero()
    assert np.all(row_indices == 0)
    assert len(col_indices) == 2


def test_labels():
    # get the texts from the training data
    examples = classify.read_smsspam("AGBIG_annotation.outt")
    labels = [label for label, _ in examples]

    # create the label encoder from the training texts
    to_labels = classify.TextToLabels(labels)

    # make sure that some sample labels are encoded as expected
    nc_index = to_labels.index("no")
    c_index = to_labels.index("yes")
    assert nc_index != c_index
    assert np.all(to_labels(["no", "yes", "yes"]) ==
                  [nc_index, c_index, c_index])


def test_prediction(capsys, min_f1=0.89, min_accuracy=0.97):
    #K FOLD TEST
    full_examples = classify.read_smsspam("AGBIG_annotation.outt")
    full_labels, full_texts = zip(*full_examples)

    clf = MLPClassifier(max_iter=1000)
    pipeline = Pipeline([
        ('vectorizer',  CountVectorizer(binary = False, ngram_range=(1, 1), max_df=1)),
        ('classifier',  AdaBoostClassifier() )])

    #print(np.asarray(train_texts[1:5]))
    k_fold = KFold(n_splits=2)

    #for LinearRegression
    #full_labels = [0 if i == "no" else i for i in full_labels]
    #full_labels = [1 if i == "yes" else i for i in full_labels]

    scores = []
    for train_indices, test_indices in k_fold.split(np.array(full_texts)):
        train_text = np.array(full_texts)[train_indices]
        train_y    = np.array(full_labels)[train_indices]

        test_text = np.array(full_texts)[test_indices]
        test_y    = np.array(full_labels)[test_indices]

        pipeline.fit(train_text, train_y)
        score = pipeline.score(test_text, test_y)
        scores.append(score)

    score = sum(scores) / len(scores)
    #KFOLD performance
    if capsys is not None:
        with capsys.disabled():
            msg = "\n{:.1%} score on MTURK development data"
            print(msg.format(score))

    '''f = open("classify.js", "w")
    porter = Porter(clf, language='js')
    output = porter.export(embed_data=True)
    f.write(output)
    f.close()'''

    #NORMAL VALIDATION
    # get texts and labels from the training data
    train_examples = classify.read_smsspam("AGBIG_annotation.outt")
    train_labels, train_texts = zip(*train_examples)

    # get texts and labels from the development data
    devel_examples = classify.read_smsspam("AGBIG_annotation.outd")
    devel_labels, devel_texts = zip(*devel_examples)

    # create the feature extractor and label encoder
    to_features = classify.TextToFeatures(train_texts)
    to_labels = classify.TextToLabels(train_labels)


    # train the classifier on the training data aka fit
    classifier = classify.Classifier()
    classifier.train(to_features(train_texts), to_labels(train_labels))

    # make predictions on the development data
    predicted_indices = classifier.predict(to_features(devel_texts))
    assert np.array_equal(predicted_indices, predicted_indices.astype(bool))

    # measure performance of predictions
    devel_indices = to_labels(devel_labels)
    spam_label = to_labels.index("yes")
    f1 = f1_score(devel_indices, predicted_indices, pos_label=spam_label)
    accuracy = accuracy_score(devel_indices, predicted_indices)

    # print out performance
    if capsys is not None:
        with capsys.disabled():
            msg = "\n{:.1%} F1 and {:.1%} accuracy on MTURK development data"
            print(msg.format(f1, accuracy))

    # make sure that performance is adequate
    #assert f1 > min_f1
    #assert accuracy > min_accuracy


@pytest.mark.xfail
def test_very_accurate_prediction():
    test_prediction(capsys=None, min_f1=0.94, min_accuracy=0.98)
