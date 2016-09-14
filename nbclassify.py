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
				return math.log1p(self.model["spamVocab"][token]/self.model["totalSpamWords"])
			else:
				return 0
		elif classification == "ham":
			if token in self.model["hamVocab"]:
				return math.log1p(self.model["hamVocab"][token]/self.model["totalHamWords"])
			else:
				return 0

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
		probabilityHam = 0
		probabilitySpam = 0
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
				#print(temp1)
				probabilityHam += temp1
				temp2 = self.getwordProbability(token,"spam")
				probabilitySpam += temp2
				if not probabilitySpam or not probabilityHam:
					print("Hey found both as zeros")
					#print(temp1)
					#print(temp2)
					#with open("temp.txt","w") as fh1:
					#	fh1.write(token)
					#	fh1.close()
					#	break
			#pS = (self.spamProbability*probabilitySpam)/(self.spamProbability*probabilitySpam + self.hamProbability*probabilityHam)
			#pH = (self.hamProbability*probabilityHam)/(self.hamProbability*probabilityHam + self.spamProbability*probabilitySpam)
			pS = math.log1p(self.spamProbability) + probabilitySpam
			pH = math.log1p(self.hamProbability)+ probabilityHam
			#print("Ham probability : "+str(pH))
			#print("Spam probability : "+str(pS))
			if pS > pH:
				print("SPAM "+filename)
				return "SPAM "+filename
			print("HAM "+filename)
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

