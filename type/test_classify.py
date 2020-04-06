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
    for example in classify.read_smsspam("AGBIGnp.out"):

        # make sure the right shape is returned
        assert len(example) == 2
        label, text = example

        # make sure the label is one of the expected two
        assert label in {"Positive", "Negative"}

        count += 1
    assert count == 1553


def test_features():
    # get the texts from the training data
    examples = classify.read_smsspam("AGBIGnp.out")
    texts = [text for _, text in examples]

    # create the feature extractor from the training texts
    to_features = classify.TextToFeatures(texts)

    # extract features for some made-up sentences
    features = to_features(["illegals should leave", "Build the wall"])
    # make sure there is one row of features for each sentence
    assert len(features.shape) == 2
    n_rows, n_cols = features.shape
    assert n_rows == 2

    # make sure there are nonzero values for some selected unigram
    # features in the first sentence
    indices = [to_features.index(f) for f in ["illegals", "wall"]]
    assert len(set(indices)) > 1
    row_indices, col_indices = features[:, indices].nonzero()
    assert np.all(row_indices == 0)
    assert len(col_indices) == 2


def test_labels():
    # get the texts from the training data
    examples = classify.read_smsspam("AGBIGnp.out")
    labels = [label for label, _ in examples]

    # create the label encoder from the training texts
    to_labels = classify.TextToLabels(labels)

    # make sure that some sample labels are encoded as expected
    #fl_index = to_labels.index("Facts/Logic")
    pt_index = to_labels.index("Positive")
    nt_index = to_labels.index("Negative")
    #a_index = to_labels.index("Affiliation")
    #h_index = to_labels.index("Humor")
    #w_index = to_labels.index("Warning")
    assert nt_index != pt_index
    #assert np.all(to_labels(["Facts/Logic", "Positive", "Negative", "Affiliation", "Humor", "Warning"]) ==
    assert np.all(to_labels(["Positive", "Negative"]) ==
                  [ pt_index, nt_index])


def test_prediction(capsys, min_f1=0.89, min_accuracy=0.97):
    #K FOLD TEST
    full_examples = classify.read_smsspam("a_lil_morenp.out")
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
        p = pipeline.predict(test_text)
        p2 = pipeline.predict_proba(test_text)
        scores.append(score)

    p_o = [j for j in p if j == "Positive"]
    p2_o = [p2[j] for j in range(0, len(p)) if p[j] == "Positive"]

#    print("Positive " , len(p_o) , " total " , len(p) , " proba " , p2_o[0])
    print("Positive " , len(p_o) , " total " , len(p) )
    
    score = sum(scores) / len(scores)
    #KFOLD performance
    if capsys is not None:
        with capsys.disabled():
            msg = "\n{:.1%} score on MTURK development data" + p_o
            print(msg.format(score))



@pytest.mark.xfail
def test_very_accurate_prediction():
    test_prediction(capsys=None, min_f1=0.94, min_accuracy=0.98)
