# Overview

This program classifies counterspeech samples through:
1. extracting n-gram features from text
2. using supervised classifiers

# Needed software 
* [git](https://git-scm.com/downloads)
* [Python (version 3.6 or higher)](https://www.python.org/downloads/)
* [numpy (version 1.11 or higher)](http://www.numpy.org/)
* [scipy (version 1.1 or higher)](https://www.scipy.org/)
* [scikit-learn (version 0.20 or higher)](http://scikit-learn.org/)
* [pytest](https://docs.pytest.org/)


# Packages used
The following objects may come in handy:
* [sklearn.feature_extraction.text.CountVectorizer]()
* [sklearn.preprocessing.LabelEncoder]()
* [sklearn.linear_model.LogisticRegression]() 


# Show results via pytest
...
collected 5 items                                                              

test_classify.py ...
89.8% F1 and 97.4% accuracy on SMSSpam development data
.x                                                   [100%]

===================== 4 passed, 1 xfailed in 1.59 seconds ======================
```
