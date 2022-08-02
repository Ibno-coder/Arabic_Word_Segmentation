import codecs
import os
from collections import Counter
from itertools import chain

def loadFromCorpus(path):
    """
    Load sentences.
    Sentences are separated by empty lines.
    returns : chars, label
    """
    allSentences = []
    sentence = []
    nbWord = 0
    allFiles = os.listdir(path)

    for file in allFiles:
        #print('loading dataset from : ', file)
        for line in codecs.open(path+'/'+file, 'r', 'utf-8'):
            line = line.rstrip()
            if not line:
                if len(sentence) > 0:
                    if 'START' not in sentence[0][0]:
                        allSentences.append(sentence)
                    sentence = []
            else:
                char = line.split()
                assert len(char) >= 2
                sentence.append(char)
                if char[0] == "WB":
                    nbWord += 1
        if len(sentence) > 0:
            if 'START' not in sentence[0][0]:
                allSentences.append(sentence)
    words, labels = zip(*[zip(*row) for row in allSentences])
    return words, labels, nbWord

def loadFromCorpus_4Seg_2(path):
    """
    Load sentences.
    Sentences are separated by empty lines.
    returns : words, label, nbWords
    """
    allSentences = []
    sentence = []
    nbWord = 0
    allFiles = os.listdir(path)

    for file in allFiles:
        #print('loading dataset from : ', file)
        for line in codecs.open(path+'/'+file, 'r', 'utf-8'):
            line = line.rstrip()
            if not line:
                if len(sentence) > 0:
                    if 'START' not in sentence[0][0]:
                        allSentences.append(sentence)
                    sentence = []
            else:
                char = line.split()
                assert len(char) >= 2
                if char[0] == "WB":
                    nbWord += 1
                else:
                    sentence.append(char)
        if len(sentence) > 0:
            if 'START' not in sentence[0][0]:
                allSentences.append(sentence)
    words, labels = zip(*[zip(*row) for row in allSentences])
    return words, labels, nbWord

def loadFromCorpus_4Seg(path):
    """
    Load sentences.
    Sentences are separated by empty lines.
    returns : words, labels, nbWords
    """
    allFiles = os.listdir(path)
    chars = []
    labels = []

    for file in allFiles:
        #print('loading dataset from : ', file)
        with codecs.open(path+'/'+file, encoding='utf-8') as fl:
            list_of_lines = [line.strip().split() for line in fl if len(line.strip().split()) == 2]
            c, trg = list(zip(*list_of_lines))
        chars += c
        labels += trg

    words = ''.join(chars).split('WB')
    nbWords = len(words)
    labels = ''.join(labels).split('WB')
    words = [tuple(word) for word in words]
    labels = [tuple(trg) for trg in labels]

    return words, labels, nbWords

def fitnessIndexTerm(chars, reserved=None, preprocess=lambda x: x):
    """
        Load list terms : chars.
        return : index to term eg. 0 : أ
    """
    if reserved is None:
        reserved = []
    allTerms = chain(*chars)
    allTerms = map(preprocess, allTerms)
    termFrequent = Counter(allTerms).most_common()  # most frequent sorted chars
    # print(termFrequent)
    index2Term = reserved + [term for term, _ in termFrequent]
    return index2Term


def buildWords(src, ref, pred):
    words = []
    rtags = []
    ptags = []
    w = ''
    r = ''
    p = ''
    for i in range(len(src)):
        if src[i] == 'WB':
            words.append(w)
            rtags.append(r)
            ptags.append(p)
            w = ''
            r = ''
            p = ''
        else:
            w += src[i]
            r += ref[i]
            p += pred[i]

    return words, rtags, ptags

def buildWords_4Seg(src, ref, pred):
    words = []
    rtags = []
    ptags = []
    l = 0
    for word, tag in zip(src, ref):
        words.append(''.join(word))
        rtags.append(''.join(tag))
        ptags.append(''.join(pred[l:l+len(word)]))
        l += len(word)
    return words, rtags, ptags

def invertIndex(index2Term):
    """
    :param index2Term: list index : term
    :return:  list inverted ( term : index )
    """
    return {term: i for i, term in enumerate(index2Term)}



def loadLookuplist(path):
    """
    Load lookp list.
    """
    listwords = {}
    for line in codecs.open(path, 'r', 'utf-8'):
        line = line.rstrip()
        listwords[line.replace('+', '')] = line
    return listwords

def getLabels(path):
    labels = []
    for line in codecs.open(path, 'r', 'utf-8'):
        line = line.strip()
        if len(line) == 0:
            continue
        splits = line.split('\t')
        labels.append(splits[1].strip())
    return list(set(labels))

