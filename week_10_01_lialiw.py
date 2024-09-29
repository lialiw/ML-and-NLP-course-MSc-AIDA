# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 00:44:27 2022

@author: Fik
"""

"""
    Sources:
        https://www.nltk.org/api/nltk.html#
        https://realpython.com/nltk-nlp-python/
        https://www.nltk.org/book/ch02.html
        https://www.nltk.org/book/ch03.html
        https://www.nltk.org/api/nltk.lm.html

        https://www.nltk.org/api/nltk.text.html
        
        https://www.tutorialspoint.com/python_text_processing/python_bigrams.htm
        https://gist.github.com/lisanka93/7b963f2ed1f3da76cfb44a97a52a82a1
        https://www.programcreek.com/python/example/86315/nltk.bigrams
        https://www.programcreek.com/python/example/59108/nltk.trigrams
        
        https://stackoverflow.com/questions/14364762/counting-n-gram-frequency-in-python-nltk/14413194
        https://stackoverflow.com/questions/27092833/unicodeencodeerror-charmap-codec-cant-encode-characters
        
"""

import nltk
from nltk.util import bigrams,trigrams
from nltk.corpus import gutenberg
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import FreqDist

from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE


def create_dictionaryOfBooks(selected_fileIds):
    setOf_words = set()
    for fileId in selected_fileIds:
        aSet = set(w.lower() for w in gutenberg.words(fileId))
        #print(len(aSet))
        setOf_words.update(aSet)
    print(len(setOf_words))
    return setOf_words


def create_dict_booksTokens(selected_fileIds):
    aDictList, aDictSet = {}, {}
    for fileId in selected_fileIds:
        rawText = gutenberg.raw(fileId)
        tokens = word_tokenize(rawText)
        #print(type(tokens), len(tokens), len(set(tokens)))
        aDictList[fileId] = tokens
        aDictSet[fileId] = set(tokens)
    return aDictList, aDictSet
      
  
def create_dict_booksSentences(selected_fileIds):
    aDictList, aDictSet = {}, {}
    for fileId in selected_fileIds:
        rawText = gutenberg.raw(fileId)
        sents = sent_tokenize(rawText)
        #print(type(sents), len(sents), len(set(sents)))
        aDictList[fileId]=sents
        aDictSet[fileId]=set(sents)
    return aDictList, aDictSet
        

def summarize_dictBooks(dict_booksLists, dict_booksSets):
    listOf_units = []
    setOf_units = set()

    for key in dict_booksLists.keys():
        listOf_units.append(dict_booksLists[key])
        
    for key in dict_booksSets.keys():
        setOf_units.update(dict_booksSets[key])
    #print(len(setOf_units))
    
    return listOf_units, setOf_units


def uni_bi_tri_grams(dict_booksTokensLists):
    aDict = {}
    
    for key in dict_booksTokensLists.keys():
        uni_fr = FreqDist(dict_booksTokensLists[key])
        bi = bigrams(dict_booksTokensLists[key])
        bi_fr = FreqDist(bi)
        tri = trigrams(dict_booksTokensLists[key])
        tri_fr = FreqDist(tri)
        aDict[key] = [uni_fr, bi_fr, tri_fr]
        
    return aDict


def writeFile_uni_bi_tri(dict_uni_bi_tri):
    fr_list =["uni", "bi", "tri"]
    
    for i in range(len(fr_list)):
        f = open(fr_list[i]+"_freq.txt","w",encoding="utf-8")
        for key in dict_uni_bi_tri.keys():
            f.write(key+'\n')
            for k,v in dict_uni_bi_tri[key][i].items():
                s='('+str(k)+','+str(v)+')\n'
                f.write(s)
            f.write('\n')
        f.close()

    
def sentences_to_words(sentences_list):
    aList = []
    for sent in sentences_list:
        sent = sent.split()
        aList.append(sent)
    return aList


def create_dict_booksSentencesToWords(dict_booksSentsLists):
    aDict={}
    for key in dict_booksSentsLists.keys():
        aDict[key] = sentences_to_words(dict_booksSentsLists[key])
    return aDict
    

def listOfStrings_to_pseudoSentense(listStrings):
    s=''
    for el in listStrings:
        s+= el+' '

    disallowed_chars = "`?_.-!()\""
    for char in disallowed_chars:
        s = s.replace(char, '')
    s = s[:-1]+'.'
    
    disallowed_pair_chars = [',.', ';.', ':.']
    for pair in disallowed_pair_chars:
        s = s.replace(pair, '.')

    s = s.replace(',\'', ',')
    s = s.capitalize()
    s = s.replace(' i ', ' I ')

    #print("skata: ",s)
    return s


def generateSents(numSents, numWords, nGram, text):
    train, vocab = padded_everygram_pipeline(nGram, text)
    lm = MLE(nGram)
    lm.fit(train, vocab)
    
    generated_sentences = []
    for i in range(numSents):
        s=''
        while(True):
            s = lm.generate(numWords)
            if '<s>' not in s and '</s>' not in s:
                break
        pseudo_s = listOfStrings_to_pseudoSentense(s)
        #print(pseudo_s)
        generated_sentences.append(pseudo_s)
    return generated_sentences


def create_dict_genSent_uni_bi_tri(dict_books_sentsToWords_Lists, numSents, numWords):
    aDict = {}
    
    for key in dict_books_sentsToWords_Lists.keys():
        aList=[]
        text = dict_books_sentsToWords_Lists[key]
        for i in range(1,4):
            generated_sentences = generateSents(numSents, numWords, i, text)
            aList.append(generated_sentences)
        aDict[key] = aList
    return aDict


def writeFile_uni_bi_tri_generatedSents(dict_uni_bi_tri):
    gram_list =["uni", "bi", "tri"]
    
    for i in range(len(gram_list)):
        f = open(gram_list[i]+"_pseudo.txt", "w", encoding="utf-8")
        for key in dict_uni_bi_tri.keys():
            f.write(key+'\n')
            for sent in dict_uni_bi_tri[key][i]:
                s = sent+'\n'; f.write(s)
            f.write('\n')
        f.close()
      
        
def main():
    
    fileIds = nltk.corpus.gutenberg.fileids(); #print(fileIds)
    selected_fileIds = ['austen-emma.txt', 'austen-persuasion.txt', 'austen-sense.txt', 'bible-kjv.txt', 'blake-poems.txt','melville-moby_dick.txt', 'milton-paradise.txt', 'shakespeare-caesar.txt', 'shakespeare-hamlet.txt', 'shakespeare-macbeth.txt']
    #print(selected_fileIds)
    #selected_fileIds = ['austen-emma.txt']
    
    gutenberg10_dictionary = create_dictionaryOfBooks(selected_fileIds)
    
    dict_booksTokensLists, dict_booksTokensSets = create_dict_booksTokens(selected_fileIds)
    #list_gutenberg10_tokens, set_gutenberg10_tokens = summarize_dictBooks(dict_booksTokensLists, dict_booksTokensSets)
    
    dict_booksSentsLists, dict_booksSentsSets = create_dict_booksSentences(selected_fileIds)
    #list_gutenberg10_sents, set_gutenberg10_sents = summarize_dictBooks(dict_booksSentsLists, dict_booksSentsSets)
    
    dict_uni_bi_tri = uni_bi_tri_grams(dict_booksTokensLists)
    writeFile_uni_bi_tri(dict_uni_bi_tri)


    dict_books_sentsToWords_Lists = create_dict_booksSentencesToWords(dict_booksSentsLists) #as Input
    numSents, numWords = 10, 10
    dict_books_generatedSentences_Lists = create_dict_genSent_uni_bi_tri(dict_books_sentsToWords_Lists, numSents, numWords)
    
    writeFile_uni_bi_tri_generatedSents(dict_books_generatedSentences_Lists)
    
if __name__ == "__main__":
    main()