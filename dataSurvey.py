import os
from sys import exit
from shutil import copy
os.chdir(os.path.abspath(os.path.dirname(__file__)))
EXIT_SUCCESS = 0
EXIT_FAILURE = 1
inputFolderPath = "."
dataSetName = "dataSet"
judgeNames = [																														\
	"distribution.py", "model.pth", "training.log", "testing.log", 																									\
	"performanceFigure" + os.sep + "testingConfusionMatrix.png", "performanceFigure" + os.sep + "trainingDetailedAccuracy.png", "performanceFigure" + os.sep + "trainingDetailedLoss.png", "performanceFigure" + os.sep + "trainingGeneralAccuracy.png", "performanceFigure" + os.sep + "trainingGeneralLoss.png", 	\
	"performanceExcel" + os.sep + "testingConfusionMatrix.xlsx", "performanceExcel" + os.sep + "testingEvaluationMatrix.xlsx", "performanceExcel" + os.sep + "testingSummaryMatrix.xlsx"													\
]
statusName = {-1:"Not a directory", 0:"Not ready", 1:"Ready", 2:"Trained"}
skippedList = ["NDER", "NSER"]


def scanFolder(dataSetPath:str) -> dict:
	dicts = {}
	if os.path.isdir(dataSetPath):
		for folder in os.listdir(dataSetPath):
			target = os.path.join(dataSetPath, folder)
			if os.path.isdir(target):
				dicts[folder] = len(os.listdir(target))
		dicts["Status"] = int(len(dicts) > 1 and (("N" in dicts and "YN" in dicts) if dataSetPath.endswith("WithEmpty") else True) and list(dicts.values())[1:] == list(dicts.values())[:-1])
	else:
		dicts["Status"] = -1
	return dicts

def isTrained(target:str) -> bool:
	for judgeName in judgeNames:
		if not os.path.isfile(os.path.join(target, judgeName)):
			print(os.path.join(target, judgeName))
			return False
	return True

def main() -> int:
	overallDicts = {}
	statusCountDict = {}
	bRet = True
	for f in os.listdir(inputFolderPath):
		target = os.path.join(os.path.join(inputFolderPath, f), dataSetName)
		if os.path.isdir(target) and f not in skippedList:
			dicts = scanFolder(target)
			if dicts["Status"] == 1 and isTrained(os.path.join(inputFolderPath, f)):
				dicts["Status"] = 2
			overallDicts[f] = dicts
			bRet = bRet and dicts["Status"] > 0
			statusCountDict.setdefault(dicts["Status"], 0)
			statusCountDict[dicts["Status"]] += 1 
	if 2 in statusCountDict and statusCountDict[2]:
		print("{0} ({1}): ".format(statusName[2], statusCountDict[2]))
		for key in list(overallDicts.keys()):
			if overallDicts[key]["Status"] == 2:
				print("{0} -> {1}".format(key, overallDicts[key]))
		print()
	if 1 in statusCountDict and statusCountDict[1]:
		print("{0} ({1}): ".format(statusName[1], statusCountDict[1]))
		for key in list(overallDicts.keys()):
			if overallDicts[key]["Status"] == 1:
				print("{0} -> {1}".format(key, overallDicts[key]))
		print()
	if 0 in statusCountDict and statusCountDict[0]:
		print("{0} ({1}): ".format(statusName[0], statusCountDict[0]))
		for key in list(overallDicts.keys()):
			if overallDicts[key]["Status"] == 0:
				print("{0} -> {1}".format(key, overallDicts[key]))
		print()
	print("Please press the enter key to exit. ")
	input()
	print()
	return EXIT_SUCCESS if bRet else EXIT_FAILURE



if __name__ == "__main__":
	exit(main())