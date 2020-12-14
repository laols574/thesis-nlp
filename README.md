# Overview

This program classifies counterspeech samples through:
1. extracting features from text
2. using supervised classifiers

## Notes

To view more information about this project, please visit http://ec2-18-219-250-45.us-east-2.compute.amazonaws.com

## Included files

* AGBIG_annotation.out - this dataset contains the annotation of counterspeech samples:|yes/no|comment|
* AGBIG_annotation.outd/AGBIG_annotation.outt - these files contain the dataset split into two sections of 3000 comments
* LICENSE - Apache license
* a_lil_more.out - I evened out the number of positive/negative samples of counterspeech to get a more even reading from the classifier
* accuracy.pdf - contains information regarding the output of the classifier with different parameters
* classify.py - this file contains the basic python for classifier function
* even.py - a small python program to test and even out the number of pos/neg samples 
* new_text.out - Apache license
* requirements.txt - contains package requirements
* test_classify.py - contains the driver for the program, including testing
* counterspeech/* - files to classify counterspeech
* type/* - files to classify different types of counterspeech (i.e. Humor)

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

## References

* CS 439 class materials 

* scikit learn documentation

# Show results via pytest -p no:warnings
...
test_classify.py .F.
57.9% score on MTURK development data

74.3% F1 and 59.2% accuracy on MTURK development data

===================== 4 passed, 1 xfailed in 1.59 seconds ======================
```

