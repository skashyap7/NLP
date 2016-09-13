#!/usr/bin/python

import argparse
import os
from os import path
import sys
import json

class nbClassify:
	def __init__(self):
		self.binaryClass = ["spam","ham"]
		self.testdataList = []
		self.model = {}
		self.spamProbability = 1
		self.hamProbability = 1
		self.classificationResult = []

	def loadModel(self, filename):
		try:
			with open(filename,"r") as fhandle:
				self.model = json.load(fhandle)
				fhandle.close()
				self.spamProbability = self.model["spamDocuments"]/self.model["totalDocuments"]
				self.hamProbability = 1 - self.spamProbability
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
				return self.model["spamVocab"][token]
			else:
				return 1
		elif classification == "ham":
			if token in self.model["hamVocab"]:
				return self.model["hamVocab"][token]
			else:
				return 1

	def classifyTestData(self):
		for f in self.testdataList:
			classification = self.classifyFile(f)
			self.classificationResult.append(classification)
		# Write the output data to a file
		with open("nboutput.txt","w") as fhandle:
			fhandle.write("\n".join(self.classificationResult))
			fhandle.close()

	def classifyFile(self, filename):
		# read a file, tokenize, calculate the probability for each class(spam/ham)
		# choose the class which has max probability for the file
		probabilityHam = 1
		probabilitySpam = 1
		doc_classification = None
		try:
			with open(filename,"r",encoding="latin1") as fhandle:
				fileContent = fhandle.read()
				fhandle.close()
			print("Evaluating test file {0}".format(filename))
			tokenizedWords = fileContent.split()
			for token in tokenizedWords:
				#print(token)
				temp1 = self.getwordProbability(token,"ham")
				print(temp1)
				probabilityHam *= temp1
				temp2 = self.getwordProbability(token,"spam")
				probabilitySpam *= temp2
				if not probabilitySpam or not probabilityHam:
					print("Hey found both as zeros")
					print(temp1)
					print(temp2)
					with open("temp.txt","w") as fh1:
						fh1.write(token)
						fh1.close()
						break
			print("--- START---")
			print(probabilityHam)
			print(probabilitySpam)
			print(self.spamProbability)
			print(self.hamProbability)
			pS = (self.spamProbability*probabilitySpam)/(self.spamProbability*probabilitySpam + self.hamProbability*probabilityHam)
			pH = (self.hamProbability*probabilityHam)/(self.hamProbability*probabilityHam + self.spamProbability*probabilitySpam)
			if pS > pH:
				print("It is SPAM")
				return "SPAM "+filename
			print("It is HAM")
			return "HAM "+filename
		except Exception as err:
			raise Exception(err)

parser = argparse.ArgumentParser(usage="python nbclassify.py <INPUTDIR>", description="Classify emails as SPAM/HAM using naive bayes model learned from labelled data")
parser.add_argument('idir', help="inputdir help")
args = parser.parse_args()

naiveBayesClassifier = nbClassify()
naiveBayesClassifier.loadModel("nbmodel.txt")
naiveBayesClassifier.getTestData(args.idir)
naiveBayesClassifier.classifyTestData()

