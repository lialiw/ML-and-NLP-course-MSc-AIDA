# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 16:33:39 2022

@author: Fik
"""

"""
    Sources:
        https://blog.chapagain.com.np/python-nltk-sentiment-analysis-on-movie-reviews-natural-language-processing-nlp/
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
        
"""
from nltk.corpus import movie_reviews, stopwords 
from random import shuffle
from nltk import FreqDist 
import string
from sklearn.model_selection import train_test_split
from nltk import NaiveBayesClassifier, classify

from nltk.tokenize import word_tokenize


def initialize_moviesCategories_shuffle_docs():
    documents = []
    for category in movie_reviews.categories():
        for fileid in movie_reviews.fileids(category):
            documents.append((movie_reviews.words(fileid), category))

    shuffle(documents)
    return documents


def freqDistr_cleanWords_all_docs():
    
    stopwords_english = stopwords.words('english')
    all_words_clean = [word.lower() for word in movie_reviews.words() if word.lower() not in string.punctuation and word.lower() not in stopwords_english]
    all_words_clean_frequency = FreqDist(all_words_clean)
    
    #print (all_words_clean[:10])
    return all_words_clean_frequency
            
            
def document_features_topN(document_words, word_features):
	# "set" function will remove repeated/duplicate tokens in the given list
	document_words = set(document_words)
	features_dict = {}
	for word in word_features:
		features_dict['contains(%s)' % word] = (word in document_words)
	return features_dict


def topN_approach(n, docs):
    
    all_words_clean_frequency = freqDistr_cleanWords_all_docs()
    #print(type(all_words_clean_frequency)); print(all_words_clean_frequency)
    
    # get 'n' frequently occuring words
    most_common_words = all_words_clean_frequency.most_common(n)
    # the most common words list's elements are in the form of tuple, get only the first element of each tuple of the word list
    word_features_most_common = [item[0] for item in most_common_words]
    
    docs_wordFeature_categ_list = []
    for (doc, category) in docs:
        docs_wordFeature_categ_list.append((document_features_topN(doc, word_features_most_common), category))
        
    return docs_wordFeature_categ_list, word_features_most_common
            

def document_features_bagOfWords(docWords):
    stopwords_english = stopwords.words('english')
    words_clean = [word.lower() for word in docWords if word.lower() not in string.punctuation and word.lower() not in stopwords_english]
    return dict([word, True] for word in words_clean)


def bagOfWords_approach():

    docs_wordFeature_categ_list = []
    for fileId in movie_reviews.fileids():
        words_clean_dict = document_features_bagOfWords(movie_reviews.words(fileId))
        docs_wordFeature_categ_list.append((words_clean_dict, movie_reviews.categories(fileId)[0]))
    return docs_wordFeature_categ_list        
        
            
def create_train_test_sets(docs_wordFeature_categ_list, trainSize):
    
    pos_docs_list = [doc for doc in docs_wordFeature_categ_list if doc[1]=='pos'] 
    #print(len(pos_docs_list))
    #print(pos_docs_list[0][1])
    neg_docs_list = [doc for doc in docs_wordFeature_categ_list if doc[1]=='neg']

    pos_docs_list_train, pos_docs_list_test = train_test_split(pos_docs_list, test_size = 1-trainSize, train_size=trainSize)
    neg_docs_list_train, neg_docs_list_test = train_test_split(neg_docs_list, test_size = 1-trainSize, train_size=trainSize) 
    
    docs_list_train = pos_docs_list_train + neg_docs_list_train; shuffle(docs_list_train)
    docs_list_test = pos_docs_list_test + neg_docs_list_test; shuffle(docs_list_test)
    
    return docs_list_train, docs_list_test
    

def train_test_model(trainSet, testSet):
    classifier = NaiveBayesClassifier.train(trainSet)
    accuracy = classify.accuracy(classifier, testSet)
    print ('Naive Bayes model\'s accuracy: ', accuracy)
    
    return classifier


def classifySingleReview(classifier, custom_review, word_features_list):  
    
    custom_review_tokens = word_tokenize(custom_review)
    if word_features_list:
        custom_review_set = document_features_topN(custom_review_tokens, word_features_list)
    else:
        custom_review_set = document_features_bagOfWords(custom_review_tokens)
    
    print ('Custom review given: ', custom_review, '\nClassified as: ',classifier.classify(custom_review_set))
 

def main():
    
    shuffled_docs = initialize_moviesCategories_shuffle_docs()#; print(type(shuffled_docs)) 
    
    topN_flag, bag_flag =  False, True
    docs_wordFeature_categ_list, word_features_list  = [], []
    if topN_flag:
        num_words = 2000
        docs_wordFeature_categ_list, word_features_list = topN_approach(num_words, shuffled_docs)
    if bag_flag:
        docs_wordFeature_categ_list = bagOfWords_approach()
    #print (docs_wordFeature_categ_list[0][1])
    #print(len(docs_wordFeature_categ_list))
    #print(type(docs_wordFeature_categ_list[0][0]))
    
    trainSize = 0.8
    docs_list_trainSet, docs_list_testSet = create_train_test_sets(docs_wordFeature_categ_list, trainSize)
    #print('train: ', len(docs_list_trainSet), ' test: ', len(docs_list_testSet))

    naiveBayes_classifier = train_test_model(docs_list_trainSet, docs_list_testSet)
    
    custom_review = "It was a wonderful and amazing movie. I loved it. Best direction, good acting."
    classifySingleReview(naiveBayes_classifier, custom_review, word_features_list)


if __name__ == "__main__": 
    main()