# -*- coding: utf-8 -*-
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
from operator import itemgetter 
import re
from nltk.stem import WordNetLemmatizer
#import nltk
from nltk import load
from os import listdir
from os.path import isfile, join
###Try nx.pagerank
import sys
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def pagerank(A, eps=.0001, d=0.85):
    P = np.ones(len(A))/len(A)
    while True:
        new_P = np.ones(len(A)) * (1 - d) / len(A) + d * A.T.dot(P)
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
    #print(sentence1)
    #print(sentence2)
    counter = CountVectorizer(stop_words=stop_words,ngram_range=(1,3))
    counter.fit([sentence1,sentence2])
    s1=counter.transform([sentence1]).toarray()
    s2=counter.transform([sentence2]).toarray()
    #all_words = list(set(sent1 + sent2))
    #vector1 = [0] * len(all_words)
    #vector2 = [0] * len(all_words)
     # build the vector for the first sentence
#==============================================================================
#     for w in sent1:
#         if w in stop_words:
#             continue
#         vector1[all_words.index(w)] += 1
#         # build the vector for the second sentence
#     for w in sent2:
#         if w in stop_words:
#                 continue
#         vector2[all_words.index(w)] += 1
#     
#==============================================================================
    
    return cosine_similarity((s1), (s2))

def build_similarity_matrix(sentences, stop_words=None):
        s = np.zeros((len(sentences),len(sentences)))
        
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i == j:
                    continue
                s[i][j] = sentence_similarity(sentences[i], sentences[j],stop_words)
                #print(s[1][j])
        for k in range(len(s)):
            if(s[k].sum() == 0):
                s[k]=0
            else:
                s[k] /= s[k].sum()
            
        return s

def textrank(sentences,og_sent, top_n=5, stop_words=None):
    s = build_similarity_matrix(sentences, stop_words)
    sentenceRanks = pagerank(s)
    #print(sentenceRanks)
    ranked_sentence_indexes = [item[0] for item in sorted(enumerate(sentenceRanks), key=lambda item: -item[1])]
    #print(ranked_sentence_indexes)
    top_n=min(len(ranked_sentence_indexes),top_n)
    selected_sentences = sorted(ranked_sentence_indexes[:top_n])    # restore original sentence order
    #print(selected_sentences)
    summary = itemgetter(*selected_sentences)(og_sent)
    #print(summary)
    print(selected_sentences)
    return summary, selected_sentences

#file=open("samp.txt","r")
def summ(file,fpath,num_sent=5):
    f=open(fpath+'/'+file,"r",encoding='utf8')
    text=f.read()
    f.close()
    sent = re.split('\. |\n|\n\n',text)
    #print(sent)
    sentences = []
    sent_orig_text = []
    for i in sent:
        words = i.lower().strip().split(' ')
        if(len(words) > 3):
            sentences.append(words)
            sent_orig_text.append(i)
    #print(sentences)
    #print(sent_orig_text)
    fname=str(file)[:-4]
    print(fname)
    fw = open("projects/MNRVA/system/"+fname+"_system.txt", 'w+')
    text=""
    for idx, sentence in enumerate(textrank(sentences,sent_orig_text,num_sent, stop_words=set(stopwords.words('english')))[0]):
        #print("%s. %s" % ((idx + 1), ' '.join(sentence)))
        text=text+sentence+'\n'
    fw.write(text)
    fw.close()

if __name__=='__main__':
    fpath=""
    try:
        fpath=str(sys.argv[1])
    except:
        fpath='articles'
    articles=listdir('./'+fpath)
    for article in articles:
        #print(article)
        summ(article,fpath)
    
