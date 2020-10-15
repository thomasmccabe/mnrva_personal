# -*- coding: utf-8 -*-
from nltk.corpus import stopwords
import numpy as np
from operator import itemgetter
import re
from nltk.stem import WordNetLemmatizer
from nltk import load
from os import listdir
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import spacy
import codecs

def pagerank(s, eps=.0001, d=0.85):
    P = np.ones(len(s))/len(s)
    while True:
        new_P = np.ones(len(s)) * (1 - d) / len(s) + d * s.T.dot(P)
        delta = abs(new_P - P).sum()
        if delta <= eps:
            return new_P
        P = new_P

def sentence_similarity(sent1, sent2, stop_words=None):
    _POS_TAGGER = 'taggers/maxent_treebank_pos_tagger/english.pickle'
    tagger = load(_POS_TAGGER)
    wnpos = lambda e: ('a' if e[0].lower() == 'j' else e[0].lower()) if e[0].lower() in ['n', 'r', 'v'] else 'n'
    lemmatizer = WordNetLemmatizer()

    if stop_words is None:
        stop_words = []
    tagged_S1=tagger.tag(sent1)
    tagged_S2=tagger.tag(sent2)

    sent1 = [lemmatizer.lemmatize(t[0],wnpos(t[1])) for t in tagged_S1]
    sent2 = [lemmatizer.lemmatize(t[0],wnpos(t[1])) for t in tagged_S2]
    sentence1=""
    sentence2=""

    for i in sent1:
        sentence1 += i + ' '
    for j in sent2:
        sentence2 += j + ' '

    counter = CountVectorizer(stop_words=stop_words, ngram_range=(1,3))
    counter.fit([sentence1, sentence2])
    s1=counter.transform([sentence1]).toarray()
    s2=counter.transform([sentence2]).toarray()
    return cosine_similarity((s1), (s2))

def build_similarity_matrix(sentencesForVectoriazation, stop_words=None):
        s = np.zeros((len(sentencesForVectoriazation), len(sentencesForVectoriazation)))

        for i in range(len(sentencesForVectoriazation)):
            for j in range(len(sentencesForVectoriazation)):
                if i == j:
                    continue
                s[i][j] = sentence_similarity(sentencesForVectoriazation[i], sentencesForVectoriazation[j],stop_words)
        for k in range(len(s)):
            if(s[k].sum() == 0):
                s[k]=0
            else:
                s[k] /= s[k].sum()
        return s

def textrank(sentencesForVectoriazation, rawTextSentences, summaryLength=5, stop_words=None):
    s = build_similarity_matrix(sentencesForVectoriazation, stop_words)
    sentenceRanks = pagerank(s)
    ranked_sentence_indexes = [item[0] for item in sorted(enumerate(sentenceRanks), key=lambda item: -item[1])]
    summaryLength = min(len(ranked_sentence_indexes),summaryLength)
    selectedSentencesPositions = sorted(ranked_sentence_indexes[:summaryLength])    # restore original sentence order
    selectedSentences = itemgetter(*selectedSentencesPositions)(rawTextSentences)
    return selectedSentences, selectedSentencesPositions

def summ(file, fpath, summaryLength=10):
    # f = open(fpath+'/'+file,"r", encoding = "ISO-8859-1")
    # f = open(fpath+'/'+file,"r", encoding = "UTF-8")
    with codecs.open(fpath+'/'+file,"r",encoding='utf-8', errors='ignore') as f:
        rawText = f.read()
        f.close()
    nlp = spacy.load('en_core_web_sm')
    doc=nlp(rawText)
    i=0
    splitSentences = []
    for sent in doc.sents:
        i=i+1
        splitSentences.append(str(sent))

    sentencesForVectoriazation = []
    rawTextSentences = []
    for i in splitSentences:
        words = i.lower().strip().split(' ')
        if(len(words) > 3):
            sentencesForVectoriazation.append(words)
            rawTextSentences.append(i)
    fname=str(file)[:-4]
    print(fname)
    summaryText= []
    selectedSentences, selectedSentencesPositions = textrank(sentencesForVectoriazation, rawTextSentences, summaryLength, stop_words=set(stopwords.words('english')))

    for idx, sentence in enumerate(selectedSentences):
        summaryText.append(sentence)
    articleLength = len(sentencesForVectoriazation)
    return summaryText, selectedSentencesPositions, articleLength

def multisumm(fpath, summaryLength):
    files = listdir('./'+fpath)
    singleDocSummaries = []

    for file in files:
        summaryText, selectedSentencesPositions, articleLength = summ(file, fpath, summaryLength)
        singleDocSummaries.append([summaryText, selectedSentencesPositions, articleLength])

    combinedSummaryText = []

    for i in singleDocSummaries:
        for j in i[0]:
            if j != '':
                combinedSummaryText.append(j)
    sentenceVectors, clusterCenters, clusterSentences = get_clusters(combinedSummaryText, summaryLength)
    rankedClusters = get_cluster_importances(clusterSentences)
    centralSentences = get_central_sentences(sentenceVectors, clusterCenters)
    return final_summ(rankedClusters, centralSentences, combinedSummaryText)

def get_clusters(combinedSummaryText, summaryLength):
    _POS_TAGGER = 'taggers/maxent_treebank_pos_tagger/english.pickle'
    tagger = load(_POS_TAGGER)
    wnpos = lambda e: ('a' if e[0].lower() == 'j' else e[0].lower()) if e[0].lower() in ['n', 'r', 'v'] else 'n'
    lemmatizer = WordNetLemmatizer()

    stop_words=set(stopwords.words('english'))

    counter = CountVectorizer(stop_words=stop_words,ngram_range=(1,3))
    counter.fit(combinedSummaryText)
    sents=[]
    sentenceVectors = []

    for sent in combinedSummaryText:
        words = sent.lower().strip().split(' ')
        tagged=tagger.tag(words)
        lem_sent_words = [lemmatizer.lemmatize(t[0],wnpos(t[1])) for t in tagged]
        lem_sentence=""
        for i in lem_sent_words:
            lem_sentence += i + ' '
        sents.append(lem_sentence)
    sentenceVectors=counter.transform(sents).toarray()
    kmeans = KMeans(n_clusters=summaryLength, random_state=0).fit(sentenceVectors)
    labels = kmeans.labels_
    clusterCenters = kmeans.cluster_centers_
    groupedSentenceVectors = []
    for j in range(1,summaryLength+1):
        clusteri = [x for i,x in enumerate(sentenceVectors) if labels[i]==j]
        groupedSentenceVectors.append(clusteri)
    clusterSentences = []
    for j in range(1,summaryLength+1):
        clusteri = [x for i,x in enumerate(sents) if labels[i]==j]
        clusterSentences.append(clusteri)
    return sentenceVectors, clusterCenters, clusterSentences

def get_cluster_importances(clusterSentences):
    cluster_sizes = []
    for cluster in clusterSentences:
        cluster_sizes.append(len(cluster))
    rankedClusters = [item[0] for item in sorted(enumerate(cluster_sizes), key=lambda item: -item[1])]
    print('=====Rank of Cluster======')
    print(rankedClusters)
    return rankedClusters

def get_central_sentences(sentenceVectors, clusterCenters):
    centralSentences = []
    closestToCentroid, _ = pairwise_distances_argmin_min(clusterCenters, sentenceVectors)
    print('=====Sentences Closest to Centroid======')
    print(closestToCentroid)
    for i in closestToCentroid:
        centralSentences.append(i)
    return centralSentences

def final_summ(rankedClusters, centralSentences, original_sents):
    output_summ = []
    for i in range(len(rankedClusters)):
        output_summ.append(original_sents[centralSentences[rankedClusters[i]]])
    return output_summ

def quotationStateLogic(doc, token, quotationState):
        if quotationState == False:
            quotationState = True
            doc[token.i].is_sent_start = True
            return doc, token, quotationState
        if quotationState == True:
            quotationState = False
            doc[token.i+1].is_sent_start = True
            return doc, token, quotationState

def set_custom_boundaries(doc):
    quotationState = False
    for token in doc[:-1]:
        if token.text == '"':
            doc, token, quotationState = quotationStateLogic(doc, token, quotationState)
        elif quotationState == True:
            doc[token.i].is_sent_start = False
    return doc

if __name__=='__main__':
    fpath=""
    summaryLength = 0
    try:
        fpath=str(sys.argv[1])
    except:
        #fpath='articles'
        fpath='small_test'
    try:
        summaryLength=int(str(sys.argv[2]))
    except:
        summaryLength = 10

    output = multisumm(fpath,summaryLength)
    for o in output:
        print(o.strip('\n'))
