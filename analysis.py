#!/usr/bin/python
import json
import argparse

class analyzeModel:
    def __init__(self):
        self.model = None
        self.threshold = 2

    def readModel(self, filename):
        with open(filename,"r") as fhandle:
            self.model = json.load(fhandle)
            fhandle.close()

    def removeCommonWords(self):
        common_elements =  list(filter(lambda x: x in self.model["hamVocab"] , self.model["spamVocab"].keys()))
        elemList = [(x,self.model["hamVocab"][x],self.model["spamVocab"][x]) for x in common_elements]
        print("Common words in both dictionaries : {0}".format(len(common_elements)))
        ignoreWords = 0
        canIgnore = []
        for elem in elemList:
            cnt1 = elem[1]
            cnt2 = elem[2]
            # check if the count of the common words is less than a threshold then
            # it is safe to remove the words
            if abs(cnt2-cnt1) <= self.threshold :
                #print(elem[0])
                canIgnore.append(elem[0])
                ignoreWords += 1
        print("It is safe to ignore {0} common words".format(ignoreWords))
        self.model["spamVocab"] = {k:v for k,v in self.model["spamVocab"].items() if k not in canIgnore}
        self.model["hamVocab"] = {k:v for k,v in self.model["hamVocab"].items() if k not in canIgnore}
        self.model["totalHamWords"] = sum(self.model["hamVocab"].values())
        self.model["totalSpamWords"] = sum(self.model["spamVocab"].values())
        self.model["vocabSize"] = self.model["vocabSize"] - ignoreWords

    def removeStopWords(self):
        stopwords = [",",".","\n",":",";"]
        print("Ignoring stop words in spam and ham dictionaries")
        self.model["spamVocab"] = {k:v for k,v in self.model["spamVocab"].items() if k not in stopwords}
        self.model["hamVocab"] = {k:v for k,v in self.model["hamVocab"].items() if k not in stopwords}
        self.model["totalHamWords"] = sum(self.model["hamVocab"].values())
        self.model["totalSpamWords"] = sum(self.model["spamVocab"].values())

    def analyzeModel(self, removecommon=None, removestopwords=None):
        if self.model:
            # get the common words and their frequency in spam and ham vocab
            if removecommon:
                self.removeCommonWords()
            if removestopwords:
                self.removeStopWords()
            # get the top ten words in spam vocab
            #self.getTop(self.model["spamVocab"])
            # get top ten words in ham vocab
            #self.getTop(self.model["hamVocab"])
            print("Rewriting the model file with ignored elements")
            with open("newmodel.txt","w") as fhandle:
                json.dump(self.model,fhandle)
                fhandle.close()

parser = argparse.ArgumentParser(usage="python analysis.py", description="Create a better model from previous model")
parser.add_argument("-m","--model", help="input model file to improve using stemming and lemmatization")
parser.add_argument("-c","--remove_common", help="improve model by removing common words with default threshold of 1",action='store_true')
parser.add_argument("-s","--remove_stopwords", help="improve model by removing stop words",action='store_true')
args = parser.parse_args()
am = analyzeModel()
if args.model:
    am.readModel(args.model)
else:
    am.readModel("nbmodel.txt")

am.analyzeModel(args.remove_common, args.remove_stopwords)