# -*- coding: utf-8 -*-
from nltk.corpus import stopwords
import numpy as np
from operator import itemgetter
from nltk.stem import WordNetLemmatizer
from nltk import load
from os import listdir
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import codecs
import math as m
from datetime import datetime
import operator
from collections import OrderedDict


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
    articlePos = []
    print(singleDocSummaries)

    for i in singleDocSummaries:
        for j in range(len(i[0])):
            if i[0][j] != '':
                combinedSummaryText.append(i[0][j])
                articlePos.append(i[1][j]/i[2])
        

    print("*****************")
    print(combinedSummaryText)
    print("*****************")
    
    print(articlePos)

    lemmatizedSummaryText = lemmatize(combinedSummaryText)
    print("lemmatizedSummaryText")
    print(lemmatizedSummaryText)
    clusterList = get_clusters(lemmatizedSummaryText)
    rankedClusters = get_cluster_importances(clusterList, lemmatizedSummaryText)
    representativeSentences = get_representative_sentences(clusterList, lemmatizedSummaryText)
    return final_summ(rankedClusters, representativeSentences, combinedSummaryText, summaryLength, articlePos)

def calc_hist_ratio(simMatrix, sentenceIndices, simThreshold):
    if sentenceIndices is None or len(sentenceIndices) <= 1:
        return 0
    aboveThreshold = 0
    belowThreshold = 0
    for i in range(0,len(sentenceIndices)):
        for j in range(i+1,len(sentenceIndices)):
            if(simMatrix[sentenceIndices[i]][sentenceIndices[j]] < simThreshold):
                belowThreshold +=1
            else:
                aboveThreshold += 1
    return aboveThreshold/(aboveThreshold+belowThreshold)

def get_clusters(lemmatizedSummaryText):
    stop_words=set(stopwords.words('english'))
    counter = CountVectorizer(stop_words=stop_words,ngram_range=(1,3))
    counter.fit(lemmatizedSummaryText)
    sentenceWordLists = []

    for sent in lemmatizedSummaryText:
        words = sent.lower().strip().split(' ')
        sentenceWordLists.append(words)
    simMatrix = build_similarity_matrix(sentenceWordLists, stop_words)

    clusterList = []
    histRatioThresh = .8
    simThreshold = .05
    eps = .05

    for i in range(0, len(sentenceWordLists)):
        placedInCluster = False
        for cluster in clusterList:
            if placedInCluster == False:
                oldClusterHR = calc_hist_ratio(simMatrix, cluster, simThreshold)
                newCluster = cluster[:]
                newCluster.append(i)
                newClusterHR = calc_hist_ratio(simMatrix, newCluster, simThreshold)
                if(newClusterHR > oldClusterHR or (newClusterHR >= histRatioThresh and (oldClusterHR - newClusterHR) <= eps)):
                    cluster.append(i)
                    print(oldClusterHR, newClusterHR)
                    placedInCluster = True
        if placedInCluster == False:
            clusterList.append([i])
    print("===clusterList from inside get_clusters function")
    print(clusterList)
    return clusterList

# def get_cluster_importances(clusterSentences):
#     cluster_sizes = []
#     for cluster in clusterSentences:
#         cluster_sizes.append(len(cluster))
#     rankedClusters = [item[0] for item in sorted(enumerate(cluster_sizes), key=lambda item: -item[1])]
#     print('=====Rank of Cluster======')
#     print(rankedClusters)
#     return rankedClusters

def uniqueWords(cluster, wordValueDictKeys):
    # This function creates a list of unique (non-duplicate) words for each cluster using sets. Called N times for N many clusters in cluserList
    nlp = spacy.load('en_core_web_sm')
    space=" "
    cluster = space.join(cluster) # Combines all sentences within a cluster into one string for processing separating them with a space
    clust = nlp(cluster)
    # Below removes stop words, punctuation and insignificant words by checkig that part of wordValueDictKeys
    words_clust = [token.text for token in clust if token.is_stop != True and token.is_punct != True and token.text in wordValueDictKeys]
    wordSet = set([]) # Set used to prevent duplicates
    for token in words_clust:
        wordSet.add(token.lower())
    uniqueWordList = list(wordSet) # Set turned into list for use within get_cluster_importances
    return uniqueWordList

def get_cluster_importances(clusterList, lemmatizedSummaryText):
    #print(clusterList)
    #print(lemmatizedSummaryText)
    clusterLemText = [] # Corresponds to clusterList - instead of #s it contains the corresponding sent. to each # from lemmatizedSummaryText
    textFromClusters = [] # List of unclustered sents. from each of the clusters to fit dict/vectorizer
    for cluster in clusterList:
        tempClustList = []
        enumeratedlemSumText = enumerate(lemmatizedSummaryText)
        for sentence in enumeratedlemSumText: # enum list looks like -> [(0,'s1'),(1,'s2')...]
            if sentence[0] in cluster: # if ref. number of elem in num list matches with a ref. number in cluster:
                tempClustList.append(sentence[1]) # append sentence associated with elem number to tempClustList
                textFromClusters.append(sentence[1])
        clusterLemText.append(tempClustList)
        tempClustList = []
    print("===clusterLemText===")
    print(clusterLemText)
    vectorizer = TfidfVectorizer()
    # Tokenize and build vocab
    vectorizer.fit(textFromClusters)
    wordValueDict = dict(zip(vectorizer.get_feature_names(), 1/vectorizer.idf_)) # Each unique word from input and it's INV TFIDF Value. Higher the value, the more occurences
    #print("===wordValueDictKeys===")
    #print(wordValueDictKeys)
    sortedWordValue = sorted(wordValueDict.items(), key=operator.itemgetter(1)) # Sorted key value pairs from least to greatest based on value, not type dict.
    # Operator.itemgetter() is faster than lambda
    #sortedWordValue = sorted(wordValueDict.items(), key=lambda kv: kv[1]) # Sorted key value pairs from least to greatest based on value, not type dict
    print("sortedWordValue")
    print(sortedWordValue)
    sortedWordValueDict = OrderedDict(sortedWordValue) # Converts sortedWordValue to type dict
    print("pre del sortedWordValueDict")
    print(len(sortedWordValueDict))
    print(sortedWordValueDict)
    delNItems = int(m.ceil(len(sortedWordValueDict)*.25)) # Number used to remove the bottom N% from sortedWordValueDict so they do not contribute toward weighting the cluster since these words are insignificant
    print("delNItems:")
    print(delNItems)
    newSortedDict = {k: sortedWordValueDict[k] for k in list(sortedWordValueDict)[delNItems:]} # Removes values from sortedWordValueDict based on delNItmes
    print("new sortedWordValueDict")
    print(len(newSortedDict))
    print(newSortedDict)
    wordValueDictKeys = [] # for uniqueWord function called in for cluster in clusterLemText loop
    #for word in wordValueDict.keys():
    for word in newSortedDict.keys():
        wordValueDictKeys.append(word)
    # Figure out how many times a word inside of a cluster occurs across all the clusters
    # Doing this for each word in a cluster and taking the sum gives the weight of each cluster
    listUniqueClusterWords = []
    for cluster in clusterLemText:
        listUniqueClusterWords.append(uniqueWords(cluster, wordValueDictKeys))
    # print("===listUniqueClusterWords===")
    # print(listUniqueClusterWords)
    clusterWeights = []
    weightThreshold = 0
    for listOfWords in listUniqueClusterWords:
        for word in listOfWords:
            clusterWeight = 0
            if dict.get(wordValueDict, word) >= weightThreshold:
                clusterWeight += m.log(1+dict.get(wordValueDict, word))
            else:
                clusterWeight += 0
        # for word in list:
        #     clusterWeight = 0
        #     try:
        #         if dict.get(wordValueDict, word) >= weightThreshold:
        #             clusterWeight += math.log(1+dict.get(wordValueDict, word))
        #         else:
        #             clusterWeight += 0
        #     except TypeError:
        #         print(word)
        clusterWeights.append(clusterWeight)
        clusterWeight = 0
    print("===ClusterWeights===")
    print(clusterWeights)
    RankedClusters = [item[0] for item in sorted(enumerate(clusterWeights), key=lambda item: -item[1])]
    print("==RankedClusters===")
    print(RankedClusters)
    return RankedClusters

def get_representative_sentences(clusterList, lemmatizedSummaryText):
    stop_words=set(stopwords.words('english'))
    representativeSentences = []
    cv = CountVectorizer(stop_words=stop_words)
    cv.fit(lemmatizedSummaryText)
    clusterVecs = []
    for cluster in clusterList:
        clusterVecs.append(cv.transform([x for i,x in enumerate(lemmatizedSummaryText) if i in cluster])) ##list of sparse matrices per sentence in cluster
    totalClusterFreqs = []
    for i in clusterVecs:
        totalClusterFreqs.append(i.sum(axis=0))
    for i in range(len(clusterList)):
        representativeSentences.append(get_rep_sentence_of_cluster(clusterList[i],lemmatizedSummaryText,totalClusterFreqs[i],totalClusterFreqs,cv))
    return representativeSentences

def get_rep_sentence_of_cluster(clusterList, lemmatizedSummaryText, clusterFreq, totalClusterFreqs, cv):
    mostRepSentence = 0
    max_weight = 0
    weight=0
    for i in clusterList:
        weight = get_representative_sentence_weight(lemmatizedSummaryText[i], clusterFreq, totalClusterFreqs, cv)
        if(weight > max_weight):
            mostRepSentence = i
            max_weight = weight
    return mostRepSentence

def get_representative_sentence_weight(sentence,clusterFreq,totalClusterFreqs,cv):
    sentWords = sentence.split(' ')
    alpha = .5
    beta = .5
    weight = 0
    for i in sentWords:
        weight += alpha*m.log10(1 + getCTF(i, clusterFreq, cv)) + beta*m.log10(1 + getCF(i,totalClusterFreqs,cv))
    return weight

def getCF(word, totalClusterFreqs, cv):
    try:
        cf = 0
        for clusterFreq in totalClusterFreqs:
            if clusterFreq[0,cv.vocabulary_[word]] >  0:
                cf +=1
        return cf
    except KeyError:
        return 0
def getCTF(word, clusterFreq, cv):
    try:
        return clusterFreq[0,cv.vocabulary_[word]]
    except KeyError:
        return 0


def final_summ(rankedClusters, centralSentences, original_sents, summaryLength, articlePos):
    print('**********FINALSUMM***********')
    print(len(rankedClusters))
    print(rankedClusters)

    print(len(centralSentences))
    print(centralSentences)

    print(len(original_sents))
    print(original_sents)
    print(summaryLength)
    
    print(len(articlePos))
    print(articlePos)
    output_summ = []
    output_summ_pos = []
    for i in range(min(summaryLength,len(rankedClusters))):
        output_summ.append(original_sents[centralSentences[rankedClusters[i]]])
        output_summ_pos.append(articlePos[centralSentences[rankedClusters[i]]])
        
    ranked_output_pos = [item[0] for item in sorted(enumerate(output_summ_pos), key=lambda item: -item[1])]
    output_summ = itemgetter(*ranked_output_pos)(output_summ)

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

def lemmatize(combinedSummaryText):
    _POS_TAGGER = 'taggers/maxent_treebank_pos_tagger/english.pickle'
    tagger = load(_POS_TAGGER)
    wnpos = lambda e: ('a' if e[0].lower() == 'j' else e[0].lower()) if e[0].lower() in ['n', 'r', 'v'] else 'n'
    lemmatizer = WordNetLemmatizer()

    lemmatizedSummaryText=[]
    for sent in combinedSummaryText:
        words = sent.lower().strip().split(' ')
        tagged=tagger.tag(words)
        lem_sent_words = [lemmatizer.lemmatize(t[0],wnpos(t[1])) for t in tagged]
        lem_sentence=""
        for i in lem_sent_words:
            lem_sentence += i + ' '
        lemmatizedSummaryText.append(lem_sentence)
    return lemmatizedSummaryText

if __name__=='__main__':
    fpath=""
    summaryLength = 0
    try:
        fpath=str(sys.argv[1])
    except:
        fpath='article_test_sets/lyft_ebike_set'
        
    try:
        summaryLength=int(str(sys.argv[2]))
    except:
        summaryLength = 5
    output = multisumm(fpath,summaryLength)
    now = datetime.now()
    timestamp = now.strftime("%m/%d/%Y-%H:%M:%S")
    fw = open("output/lyft_ebike_summary_01.txt", 'w+')
    outputText = ""
    for o in output:
        outputText += o + '\n'
    fw.write(outputText)
    fw.close()