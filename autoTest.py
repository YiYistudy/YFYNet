import os
from sys import exit
from shutil import copy
os.chdir(os.path.abspath(os.path.dirname(__file__)))
EXIT_SUCCESS = 0
EXIT_FAILURE = 1
inputFolderPath = "."
sourceScriptPath = "distribution.py"
scriptName = "distribution.py"
judgeName = "model.pth"
commentSymbols = ["#", "@", "%", "//", "__"]
skippedList = ["NDER", "NSER"]
parameter = "q"
doRetraining = False


def isCommented(f:str) -> bool:
	for commentSymbol in commentSymbols:
		if f.startswith(commentSymbol):
			return True
	return False

def main():
	for f in os.listdir(inputFolderPath):
		target = os.path.join(inputFolderPath, f)
		if os.path.isdir(target) and f not in skippedList:
			scriptPath = os.path.join(target, scriptName)
			try:
				copy(sourceScriptPath, scriptPath)
			except Exception as e:
				print("\"{0}\" -> {1}".format(scriptPath, e))
	print("*** Finished sending \"{0}\" to all subfolders. ***".format(sourceScriptPath))
	for f in os.listdir(inputFolderPath):
		target = os.path.join(inputFolderPath, f)
		if os.path.isdir(target) and f not in skippedList and not isCommented(f):
			scriptPath = os.path.join(target, scriptName)
			judgePath = os.path.join(target, judgeName)
			if os.path.isfile(scriptPath) and (not os.path.isfile(judgePath) or doRetraining):
				print("*** \"{0}\" has started. ***".format(scriptPath))
				try:
					os.system("\"{0}\" {1}".format(scriptPath, parameter))
				except:
					pass
				print("*** \"{0}\" has finished. ***".format(scriptPath))
	print("*** All are finished. Please press the enter key to exit. ***")
	input()
	return EXIT_SUCCESS
			


if __name__ == "__main__":
	exit(main())