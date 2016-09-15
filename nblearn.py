#!/usr/bin/python

import argparse
import os
from os import path
import collections
import json
import time
import re

class nbLearner:
	def __init__(self):
		self.spamFileList = []
		self.hamFileList = []
		self.spamVocab = collections.defaultdict(int)
		self.hamVocab = collections.defaultdict(int)
		self.vocabSize = 0
		self.classificationModel = {}

	def generateClassedDataListing(self, dirname):
		patternSpam = re.compile(".*.spam.txt")
		for root, dirs, files in os.walk(dirname):
			# get the directory name (SPAM or HAM)
			for sfile in files:
				classification = patternSpam.match(sfile)
				if classification:
					self.spamFileList.append(os.path.join(root,sfile))
				else:
					self.hamFileList.append(os.path.join(root,sfile)) 
	
	def extractFeaturesforTaggedData(self, classification):
		# read each file from the processed list and extract features
		# update the feature counts into the hash based on the classification
		if classification == "spam":
			targetVocab = self.spamVocab
			classifiedList = self.spamFileList
		else:
			targetVocab = self.hamVocab
			classifiedList = self.hamFileList
		for f in classifiedList:
			with open(f,"r",encoding="latin1") as fhandle:
				fileContent = fhandle.read()
				fhandle.close()
			tokenizedWords = fileContent.split()
			for token in tokenizedWords:
				targetVocab[token] += 1
		
	def genClassificationModel(self):
		self.classificationModel["totalDocuments"] = len(self.hamFileList) + len(self.spamFileList)
		self.classificationModel["spamDocuments"] = len(self.spamFileList)
		self.classificationModel["hamDocuments"] = len(self.hamFileList)
		self.classificationModel["totalSpamWords"]	= sum(self.spamVocab.values())
		self.classificationModel["totalHamWords"] = sum(self.hamVocab.values())
		self.classificationModel["vocabSize"] = self.getVocabSize()
		#for key in self.hamVocab:
		#	self.hamVocab[key] /= self.classificationModel["totalHamWords"]
		#for key in self.spamVocab:
		#	self.spamVocab[key] /= self.classificationModel["totalSpamWords"]
		self.classificationModel["hamVocab"] = self.hamVocab
		self.classificationModel["spamVocab"] = self.spamVocab
		#print(self.classificationModel)
		# write model data to file
		with open("nbmodel.txt","w+") as fhandle:
			json.dump(self.classificationModel,fhandle, indent=4)
			fhandle.close()

	def getVocabSize(self):
		common_elements =  len(list(filter(lambda x: x in self.hamVocab.keys(), self.spamVocab.keys())))
		return (len(self.hamVocab.keys()) + len(self.spamVocab.keys()) - common_elements)

start_time = time.time()
parser = argparse.ArgumentParser(usage="python nblearn.py <INPUTDIR>", description="Learn a naive bayes model from labelled data")
parser.add_argument('idir', help="inputdir help")
args = parser.parse_args()

naiveBayesLearner = nbLearner()
naiveBayesLearner.generateClassedDataListing(args.idir)
naiveBayesLearner.extractFeaturesforTaggedData("spam")
naiveBayesLearner.extractFeaturesforTaggedData("ham")
naiveBayesLearner.genClassificationModel()
print("Took {0} seconds to execute".format(time.time()-start_time))
#print(naiveBayesLearner.hamVocab)
#print(sorted(naiveBayesLearner.spamVocab.keys()))
#print(naiveBayesLearner.getVocabSize())
#print(args)