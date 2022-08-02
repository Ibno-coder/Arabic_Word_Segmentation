import codecs
import re
from nltk.tokenize import word_tokenize
# tokenization a text clair ( no diacritics )

def tokenizationText(txtClair):
    Tokens = []
    # remove Diacritics if exist
    txtClair = re.sub(u'[ـًٌٍَُِّْ]', '', txtClair)

    # split on delimiters after on white space
    for allWord in re.split(u'[،:.]', txtClair):
        for word in allWord.split():
            for elt in word_tokenize(word):
                    Tokens.append(elt)
    return Tokens


def loadLblRewerite(path):
    return set(line.rstrip() for line in codecs.open(path, 'r', 'utf-8'))

def splitText_withSen(inptText):
    sentence = []
    if len(inptText) < 2:
        print("Error input")
        return False
    words = tokenizationText(inptText)  # words in one line
    for word in words:
        sentence.append(list(word))

    return sentence

