# Objectives

The learning objectives of this assignment are to:
1. practice extracting n-gram features from text
2. become familiar with training supervised classifiers

# Setup your environment

You will need to set up an appropriate coding environment on whatever computer
you expect to use for this assignment.
Minimally, you should install:

* [git](https://git-scm.com/downloads)
* [Python (version 3.6 or higher)](https://www.python.org/downloads/)
* [numpy (version 1.11 or higher)](http://www.numpy.org/)
* [scipy (version 1.1 or higher)](https://www.scipy.org/)
* [scikit-learn (version 0.20 or higher)](http://scikit-learn.org/)
* [pytest](https://docs.pytest.org/)

# Check out a new branch

Before you start editing any code, you will need to create a new branch in your
GitHub repository to hold your work.

1. Go to the repository that GitHub Classroom created for you,
`https://github.com/UA-LING-439-SP19/ngram-classifier-<your-username>`, where
`<your-username>` is your GitHub username, and
[create a branch through the GitHub interface](https://help.github.com/articles/creating-and-deleting-branches-within-your-repository/).
You may name the branch anything you like.
2. Clone the repository to your local machine and checkout the branch you
just created:
   ```
   git clone -b <branch> https://github.com/UA-LING-439-SP19/ngram-classifier-<your-username>.git
   ```
   where `<branch>` is whatever you named your branch.


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
