#!/usr/bin/python

import argparse
import os
from os import path
import sys
import json
import math

class nbClassify:
	def __init__(self):
		self.binaryClass = ["spam","ham"]
		self.testdataList = []
		self.model = {}
		self.spamProbability = 1
		self.hamProbability = 1
		self.classificationResult = []
		self.spamcount = 0
		self.hamcount = 0
		self.spamtest = 0
		self.hamtest = 0

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
			for subdir in dirs:
				for root1, subsubdirs, samplefiles in os.walk(os.path.join(root,subdir)):
					for sfile in samplefiles:
						self.testdataList.append(os.path.join(root1,sfile))
	
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
			#print("Evaluating test file {0}".format(filename))
			tokenizedWords = fileContent.split()
			for token in tokenizedWords:
				#print(token)
				temp1 = 0
				temp2 = 0
				temp1 = self.getwordProbability(token,"ham")
				#print(temp1)
				probabilityHam += temp1
				temp2 = self.getwordProbability(token,"spam")
				probabilitySpam += temp2
			pS = math.log1p(self.spamProbability) + probabilitySpam
			pH = math.log1p(self.hamProbability) + probabilityHam
			#print("Ham probability : "+str(pH))
			#print("Spam probability : "+str(pS))
			if pS >= pH:
				print("SPAM "+filename)
				self.spamcount += 1
				return "SPAM "+filename
			else:
				print("HAM "+filename)
				self.hamcount += 1
				return "HAM "+filename
		except Exception as err:
			raise Exception(err)

	def classifyTestData(self):
		for f in self.testdataList:
			dirName = f.split("\\")[6]
			if dirName == "spam":
				self.spamtest += 1
			elif dirName == "ham":
				self.hamtest += 1
			classification = self.classifyFile(f)
			self.classificationResult.append(classification)
		# Write the output data to a file
		with open("nboutput.txt","w") as fhandle:
			fhandle.write("\n".join(self.classificationResult))
			fhandle.close()

parser = argparse.ArgumentParser(usage="python nbclassify.py <INPUTDIR>", description="Classify emails as SPAM/HAM using naive bayes model learned from labelled data")
parser.add_argument('idir', help="inputdir help")
args = parser.parse_args()

naiveBayesClassifier = nbClassify()
naiveBayesClassifier.loadModel("nbmodel.txt")
naiveBayesClassifier.getTestData(args.idir)
naiveBayesClassifier.classifyTestData()
print("Total no. of SPAM found by classifier : {0}".format(naiveBayesClassifier.spamcount))
print("Total no. of SPAM in test data : {0}".format(naiveBayesClassifier.spamtest))
print("Total no. of HAM found by classifier : {0} ".format(naiveBayesClassifier.hamcount))
print("Total no. of HAM in test data : {0}".format(naiveBayesClassifier.hamtest))

