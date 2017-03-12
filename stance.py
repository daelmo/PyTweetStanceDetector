from nltk.tokenize import TweetTokenizer
import pandas
import codecs
from nltk.corpus import stopwords
import numpy as np
#from sklearn.neighbors.nearest_centroid import NearestCentroid





topic = "Atheism"
filePath = "train.txt"


def main():
    # read File
    doc = codecs.open(filePath, 'r', 'UTF-8')
    df = pandas.read_csv(doc, sep='\t')

    #tokenize
    tweetsTokenList = []
    tupleVocabularySet = set()
    tknzr = TweetTokenizer()
    for index, row in df.iterrows():
        tweetTokens = tknzr.tokenize(row['Tweet'])

        stopwordList = set(stopwords.words('english'))
        tweetTokens =  [i.lower() for i in tweetTokens if i not in stopwordList and len(i) > 1]
        tweetsTokenList.append(tweetTokens)

        # build tupleVocabulary
        for word1, word2 in zip(tweetTokens[:-1], tweetTokens[1:]):
            tupleVocabularySet.add((word1, word2))


    #create vectorMatrix with bigram count
    tweetCount = len(df.index)
    bigramCount = len(tupleVocabularySet)
    tupleVocabularyList = list(tupleVocabularySet)

    bigramMatrix = np.zeros(shape=(bigramCount, tweetCount))

    rowIndex = 0

    for tweetTokens in tweetsTokenList:

        for word1, word2 in zip(tweetTokens[:-1], tweetTokens[1:]):
            position = tupleVocabularyList.index((word1, word2))
            bigramMatrix[position][index] += 1
        rowIndex += 1

    print bigramMatrix


    #train classifier

    #actualStances = np.array(df["Stance"])
    #clf = NearestCentroid()
    #clf.fit(bigrammMatrix, actualStances)



main()