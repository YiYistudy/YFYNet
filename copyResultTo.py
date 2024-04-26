import os
from sys import exit
from shutil import copy
try:
	from numpy import all as npAll
	from cv2 import imread, imwrite
	def autoCropFigure(figureFp:str, offset:int = 0) -> bool:
		try:
			img = imread(figureFp)
			if img.ndim in (2, 3):
				left = 0
				while npAll(img[:, left] == [255, 255, 255]):
					left += 1
				top = 0
				while npAll(img[top] == [255, 255, 255]):
					top += 1
				right = img.shape[1] - 1
				while npAll(img[:, right] == [255, 255, 255]):
					right -= 1
				bottom = img.shape[0] - 1
				while npAll(img[bottom] == [255, 255, 255]):
					bottom -= 1
				imwrite(figureFp, img[max(0, top - offset):min(img.shape[0], bottom + 1 + offset), max(0, left - offset):min(img.shape[1], right + 1 + offset)])
				print("[V] Crop \"{0}\" successfully: (l, t, r, b) = {1} -> {2}. ".format(figureFp, (0, 0, img.shape[1] - 1, img.shape[0] - 1), (max(0, left - offset), max(0, top - offset), min(img.shape[1], right + offset), min(img.shape[0], bottom + offset))))
				return True
			else:
				print("[X] Failed cropping \"{0}\": The image does not meet ``img.ndim in (2, 3)``. ".format(figureFp))
		except Exception as e:
			print("[X] Failed cropping \"{0}\": {1}".format(figureFp, e))
			return False
except:
	def autoCropFigure(figureFp:str, offset:int = 1) -> bool:
		return False
os.chdir(os.path.abspath(os.path.dirname(__file__)))
EXIT_SUCCESS = 0
EXIT_FAILURE = 1
inputFolderPath = "."
resultNames = [																														\
	"model.pth", "training.log", "testing.log", 																									\
	"performanceFigure" + os.sep + "testingConfusionMatrix.png", "performanceFigure" + os.sep + "trainingDetailedAccuracy.png", "performanceFigure" + os.sep + "trainingDetailedLoss.png", "performanceFigure" + os.sep + "trainingGeneralAccuracy.png", "performanceFigure" + os.sep + "trainingGeneralLoss.png", 	\
	"performanceExcel" + os.sep + "testingConfusionMatrix.xlsx", "performanceExcel" + os.sep + "testingEvaluationMatrix.xlsx", "performanceExcel" + os.sep + "testingSummaryMatrix.xlsx"													\
]
commentSymbols = ["#", "@", "%", "//", "__"]
skippedList = ["NDER", "NSER"]
doCropping = True
cropOffset = 30
figureExts = [".jpeg", ".jpg", ".png", ".tiff"]


def isCommented(f:str) -> bool:
	for commentSymbol in commentSymbols:
		if f.startswith(commentSymbol):
			return True
	return False

def handleFolder(folder:str) -> bool:
	if folder in ("", ".", "./", ".\\"):
		return True
	elif os.path.exists(folder):
		return os.path.isdir(folder)
	else:
		try:
			os.makedirs(folder)
			return True
		except:
			return False

def copyTo(sourceFp:str, targetFp:str) -> int:
	if os.path.isfile(sourceFp):
		if handleFolder(os.path.split(targetFp)[0]):
			try:
				copy(sourceFp, targetFp)
				if os.path.splitext(sourceFp)[1].lower() in figureExts and doCropping:
					autoCropFigure(targetFp, cropOffset)
				print("[V] \"{0}\" -> \"{1}\"".format(sourceFp, targetFp))
				return 1
			except Exception as e:
				print("[X] \"{0}\" -> \"{1}\": {2}".format(sourceFp, targetFp, e))
				return 0
		else:
			print("[X] \"{0}\" -> \"{1}\": The parent folder is not created successfully. ".format(sourceFp, targetFp))
			return 0
	else:
		return -1

def main() -> int:
	targetFolderPath = input("Please enter the target folder: ")
	successCount, totalCount = 0, 0
	for source in os.listdir(inputFolderPath):
		sourceFolder = os.path.join(inputFolderPath, source)
		if os.path.isdir(sourceFolder) and source not in skippedList and not isCommented(source):
			for name in resultNames:
				bRet = copyTo(os.path.join(sourceFolder, name), os.path.join(os.path.join(targetFolderPath, source), name))
				if bRet >= 0:
					totalCount += 1
					if bRet == 1:
						successCount += 1
	if totalCount:
		print("{0} / {1} = {2}%".format(successCount, totalCount, 100 * successCount / totalCount))
	else:
		print("Nothing was handled. ")
	print("Please press the enter key to exit. ")
	input()
	return EXIT_SUCCESS if bRet else EXIT_FAILURE



if __name__ == "__main__":
	exit(main())