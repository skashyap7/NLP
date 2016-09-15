#!/usr/bin/python

import argparse
import os
from os import path
import sys
import json
import math
import re

class nbClassify:
	def __init__(self):
		self.binaryClass = ["spam","ham"]
		self.testdataList = []
		self.model = {}
		self.spamProbability = 1
		self.hamProbability = 1
		self.actualHamData = []
		self.actualSpamData = []
		self.classifiedHamData = []
		self.classifiedSpamData = []
		self.classificationResult = []

	def loadModel(self, filename):
		try:
			with open(filename,"r") as fhandle:
				self.model = json.load(fhandle)
				fhandle.close()
				self.spamProbability = self.model["spamDocuments"]/self.model["totalDocuments"]
				self.hamProbability = self.model["hamDocuments"]/self.model["totalDocuments"]
		except FileNotFoundError:
			print("Model file not found ! Please ensure nbmodel.txt is present")
			sys.exit()

	def getTestData(self, dirname):
		for root, dirs, files in os.walk(dirname):
			for f in files:
				#print(os.path.join(root,f))
				self.testdataList.append(os.path.join(root,f))
	
	def getwordProbability(self, token, classification):
		if classification == "spam":
			if token in self.model["spamVocab"]:
				cnt = self.model["spamVocab"][token]
			else:
				cnt =  0
			return math.log(( cnt + 1)/(self.model["totalSpamWords"] + self.model["vocabSize"]),2)
		elif classification == "ham":
			if token in self.model["hamVocab"]:
				cnt = self.model["hamVocab"][token]
			else:
				cnt = 0
			return math.log(( cnt  + 1)/(self.model["totalHamWords"] + self.model["vocabSize"]), 2) 

	def classifyFile(self, filename):
		# read a file, tokenize, calculate the probability for each class(spam/ham)
		# choose the class which has max probability for the file
		probabilityHam = 0
		probabilitySpam = 0
		doc_classification = None
		try:
			with open(filename,"r",encoding="latin1") as fhandle:
				fileContent = fhandle.read()
				fhandle.close()
			tokenizedWords = fileContent.split()
			for token in tokenizedWords:
				temp1 = 0
				temp2 = 0
				temp1 = self.getwordProbability(token,"ham")
				probabilityHam += temp1
				temp2 = self.getwordProbability(token,"spam")
				probabilitySpam += temp2
			pS = math.log1p(self.spamProbability) + probabilitySpam
			pH = math.log1p(self.hamProbability) + probabilityHam
			if pS >= pH:
				self.classifiedSpamData.append(filename)
				return "SPAM "+filename
			else:
				self.classifiedHamData.append(filename)
				return "HAM "+filename
		except Exception as err:
			raise Exception(err)

	def classifyTestData(self):
		patternSpam = re.compile(".*.spam.txt")
		#print(len(self.testdataList))
		self.actualSpamData = [x for x in self.testdataList if patternSpam.match(x)]
		self.actualHamData = [x for x in self.testdataList if not patternSpam.match(x)]
		for f in self.testdataList:
			classification = self.classifyFile(f)
			self.classificationResult.append(classification)
		# Write the output data to a file
		with open("nboutput.txt","w") as fhandle:
			fhandle.write("\n".join(self.classificationResult))
			fhandle.close()
	
	def getStats(self):
		print("Total no. of SPAM found by classifier : {0}".format(len(nbc.classifiedSpamData)))
		print("Total no. of SPAM in test data : {0}".format(len(nbc.actualSpamData)))
		print("Total no. of HAM found by classifier : {0} ".format(len(nbc.classifiedHamData)))
		print("Total no. of HAM in test data : {0}".format(len(nbc.actualHamData)))
		
		# hams which were not classifed as hams
		nHam = len([x for x in nbc.actualHamData if x not in nbc.classifiedHamData])
		# spams not classified as spams
		nSpam = len([x for x in nbc.actualSpamData if x not in nbc.classifiedSpamData])
		
		print("Spams not identified as spams: "+str(nSpam))
		print("Hams not identified as hams: "+str(nHam))
		
		total = nbc.model["totalDocuments"]
		accuracy = (total - (nHam + nSpam))/total
		precision_spam = (len(nbc.classifiedSpamData) - nHam)/len(nbc.classifiedSpamData)
		precision_ham = (len(nbc.classifiedHamData) - nSpam)/len(nbc.classifiedHamData)
		recall_spam = (len(nbc.classifiedSpamData) - nHam)/(len(nbc.classifiedSpamData) - nHam + nSpam)
		recall_ham = (len(nbc.classifiedHamData) - nHam)/(len(nbc.classifiedHamData) - nSpam + nHam)
		fscore_spam =  2*precision_spam*recall_spam/(precision_spam+ recall_spam)
		fscore_ham =  2*precision_ham*recall_ham/(precision_ham+ recall_ham)
		
		print("Accuracy is : {0}".format(accuracy)) 
		print("Precision of Spam: {0}".format(precision_spam))
		print("Precision of Ham: {0}".format(precision_ham))
		print("Recall of Spam : {0}".format(recall_spam))
		print("Recall of Ham : {0}".format(recall_ham))
		print("F-Score for Spam : {0}".format(fscore_spam))
		print("F-Score for Ham : {0}".format(fscore_ham))

parser = argparse.ArgumentParser(usage="python nbclassify.py <INPUTDIR>", description="Classify emails as SPAM/HAM using naive bayes model learned from labelled data")
parser.add_argument('idir', help="inputdir help")
parser.add_argument('-m','--model', help="model file to load")
args = parser.parse_args()

nbc = nbClassify()
if not args.model:
	nbc.loadModel("nbmodel.txt")
else:
	nbc.loadModel(args.model)
nbc.getTestData(args.idir)
nbc.classifyTestData()
# Comment the below line to not print stats
nbc.getStats()

