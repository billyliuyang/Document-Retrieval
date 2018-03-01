# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 09:29:52 2016

@author: BillyLiu
"""
import sys,re,getopt
import math

from nltk.stem import PorterStemmer
from collections import Counter
from read_documents import ReadDocuments

"""\
------------------------------------------------------------
OPTIONS:
    -s FILE: the stoplist
    -i FILE : output file 
    -n NUMBER : retrieving for a specific query
    -I : weather to use stemmer
    -A : retrieving for the full query set
------------------------------------------------------------
"""
opts, args = getopt.getopt(sys.argv[1:],'Is:Ai:n:')
opts = dict(opts)

filename = args[0]
queryname = args[1]
#collect stoplist
stops = set()
if '-s' in opts:
    with open(opts['-s'],'r') as stop_fs:
        for line in stop_fs :
            stops.add(line.strip())

#output file
if '-i' in opts:
    f = open(opts['-i'],'w')

#Filter the document
def filter_document(filename):
    documents = ReadDocuments(filename)
    wordRE = re.compile(r'[A-Za-z]+')
    stemmer = PorterStemmer()
    filtereddoc = []
    for doc in documents:
        doclist = []
        for line in doc.lines:
            line.replace("\n","")
            for word in wordRE.findall(line.lower()):
                if word not in stops:
                    if '-I' in opts:
                        word = stemmer.stem(word)
                    doclist.append(word)
        c = Counter(doclist)
        filtereddoc.append(c)
    return filtereddoc            


#calculate idf                   
def tf(word, count):
    return count[word] / sum(count.values())
    
def n_containing(word, count_list):
    return sum(1 for count in count_list if word in count)

def idf(word, count_list):
    return math.log(len(count_list) / (1 + n_containing(word, count_list)))
    
def tfidf(word, count, count_list):
    return tf(word, count) * idf(word, count_list)

#filter documents and calculate tf-idf for each document
countlist = filter_document(filename)
doctfidf = []
for i, count in enumerate(countlist):
    scores = {word: tfidf(word, count, countlist) for word in count}
    doctfidf.append(scores)

#filter queries and calculate tf-idf for each query
querylist = filter_document(queryname)
quetfidf = []
for i, count in enumerate(querylist):
    scores = {word: tfidf(word, count, querylist) for word in count}
    quetfidf.append(scores)

#retrieving all queries
if '-A' in opts:
    for i,query in enumerate(quetfidf):
        sims = {}
        for j,doc in enumerate(doctfidf):
            si={}
            for word in quetfidf[i]:
                if word in doctfidf[j]:
                    si[word] = 1
            if len(si) == 0:
                sims[j] = 0
                continue
            pSum = sum([quetfidf[i][word]*doctfidf[j][word] for word in si])
            sum1Sq = sum([pow(quetfidf[i][word],2) for word in si])
            sum2Sq = sum([pow(doctfidf[j][word],2) for word in si])
            den = math.sqrt(sum1Sq*sum2Sq)
            if den == 0:
                sims[j] = 0
                continue
            r = float(pSum)/den
            sims[j] = r
        sorted_sims = sorted(sims.items(),key=lambda x: x[1],reverse=True)
        for k,v in sorted_sims[:5]:
                f.write(str(i+1)+'\t')
                f.write(str(k+1)+'\n')

#retrieving a specific query
if '-n' in opts:
    sims = {}
    queryNum = int(opts['-n'])-1
    for j,doc in enumerate(doctfidf):
        si={}
        for word in quetfidf[queryNum]:
            if word in doctfidf[j]:
                si[word] = 1
        if len(si) == 0:
            continue
        pSum = sum([quetfidf[queryNum][word]*doctfidf[j][word] for word in si])
        sum1Sq = sum([pow(quetfidf[queryNum][word],2) for word in si])
        sum2Sq = sum([pow(doctfidf[j][word],2) for word in si])
        den = math.sqrt(sum1Sq*sum2Sq)
        if den == 0:
            continue
        r = float(pSum)/den
        if r == 0:
            continue
        sims[j] = r
    sorted_sims = sorted(sims.items(),key=lambda x: x[1],reverse=True)
    for k,v in sorted_sims[:]:
        f.write(str(queryNum+1)+'\t')
        f.write(str(k+1)+'\n')
    
print("done")
        
                
        
                             
                    
                    
                    