from nltk.tokenize import TweetTokenizer
import pandas
import codecs
from nltk.corpus import stopwords
import numpy as np
from sklearn.neighbors.nearest_centroid import NearestCentroid


topic = "Climate Change is a Real Concern"
pathTrainingSet = "train.txt"
pathTestSet = "test.txt"
stopwordList = set(stopwords.words('english'))
vocabularyTupleSet = set()
tknzr = TweetTokenizer()
classifier = None

def main():
    classifier = initClassifier()
    matrix = calculateTestMatrix()
    print classifier.predict(matrix)


def initClassifier():
    # read File
    doc = codecs.open(pathTrainingSet, 'r', 'UTF-8')
    df = pandas.read_csv(doc, sep='\t')

    # cut to topic
    df = df[df.Target == topic]

    #tokenize
    tweetsTokenList = []

    for index, row in df.iterrows():
        tweetTokens = tokenize(row["Tweet"])
        tweetsTokenList.append(tweetTokens)

        # build tupleVocabulary
        for word1, word2 in zip(tweetTokens[:-1], tweetTokens[1:]):
            vocabularyTupleSet.add((word1, word2))


    #create vectorMatrix with bigram count
    tweetCount = len(df.index)
    bigramCount = len(vocabularyTupleSet)
    vocabularyTupleList = list(vocabularyTupleSet)

    bigramMatrix = np.zeros(shape=(tweetCount, bigramCount))

    rowIndex = 0

    for tweetTokens in tweetsTokenList:

        for word1, word2 in zip(tweetTokens[:-1], tweetTokens[1:]):
            position = vocabularyTupleList.index((word1, word2))
            bigramMatrix[rowIndex][position] += 1
        rowIndex += 1

    #train classifier
    actualStances = np.array(df["Stance"])
    clf = NearestCentroid()
    clf.fit(bigramMatrix, actualStances)
    return clf

def calculateTestMatrix():
    # read File
    doc = codecs.open(pathTrainingSet, 'r', 'UTF-8')
    df = pandas.read_csv(doc, sep='\t')

    # cut to topic
    df = df[df.Target == topic]

    # tokenisation of target tweets
    tweetsTokenList = []

    for index, row in df.iterrows():
        tweetTokens = tokenize(row["Tweet"])
        tweetsTokenList.append(tweetTokens)

     # create vectorMatrix with bigram count
    tweetCount = len(df.index)
    bigramCount = len(vocabularyTupleSet)
    tupleVocabularyList = list(vocabularyTupleSet)

    bigramMatrix = np.zeros(shape=(tweetCount, bigramCount))

    rowIndex = 0

    for tweetTokens in tweetsTokenList:

        for word1, word2 in zip(tweetTokens[:-1], tweetTokens[1:]):
            position = tupleVocabularyList.index((word1, word2))
            bigramMatrix[rowIndex][position] += 1
        rowIndex += 1

    return bigramMatrix


def tokenize(tweet):
    tweetTokens = tknzr.tokenize(tweet)
    tweetTokens = [i.lower() for i in tweetTokens if i not in stopwordList and len(i) > 1]
    return tweetTokens



main()