from nltk.tokenize import TweetTokenizer
import pandas
import codecs
import nltk
from nltk.corpus import stopwords





topic = "Atheism"
filePath = "train.txt"


def main():
    # read File
    doc = codecs.open(filePath, 'r', 'UTF-8')
    df = pandas.read_csv(doc, sep='\t')

    #tokenize
    tokenized = []
    tupleVocabulary = set()
    tknzr = TweetTokenizer()
    for index, row in df.iterrows():
        wordList = tknzr.tokenize(row['Tweet'])

        stop = set(stopwords.words('english'))
        wordList =  [i.lower() for i in wordList if i not in stop and len(i) > 1]
        tokenized.append(wordList)

        # build tupleVocabulary
        for word1, word2 in zip(wordList[:-1], wordList[1:]):
            tupleVocabulary.add((word1, word2))




    print tupleVocabulary.__sizeof__()





main()