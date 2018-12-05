import pandas as pd
import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt
############################################################################################################################
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import itertools
############################################################################################################################
from lime import lime_text
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer
from nltk.corpus import stopwords


def get_metrics(y_test, y_predicted):  
    # true positives / (true positives+false positives)
    precision = precision_score(y_test, y_predicted, pos_label=None,
                                    average='weighted')             
    # true positives / (true positives + false negatives)
    recall = recall_score(y_test, y_predicted, pos_label=None,
                              average='weighted')
    
    # harmonic mean of precision and recall
    f1 = f1_score(y_test, y_predicted, pos_label=None, average='weighted')
    
    # (true positives + true negatives) / total
    accuracy = accuracy_score(y_test, y_predicted)
    return accuracy, precision, recall, f1

'''
confusion matrix visulization
'''


def tfidf(data):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    train = tfidf_vectorizer.fit_transform(data)

    return train, tfidf_vectorizer