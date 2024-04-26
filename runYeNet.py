import os
from sys import argv, exit
from ast import literal_eval
from random import sample as randomSample
from re import findall
from time import time
try:
	from numpy import array, sum as npSum
	from matplotlib import pyplot as plt
	from matplotlib.ticker import MaxNLocator
	try:
		from imageio.v3 import imread
	except:
		try:
			from imageio.v2 import imread
		except:
				from imageio import imread
	from PIL.Image import fromarray
	from torch import cat, device as torchDevice, float as torchFloat, from_numpy, load as torchLoad, long, no_grad, save as torchSave, tensor
	from torch.cuda import is_available
	from torch.nn import AvgPool2d, Conv2d, Hardtanh, Linear, LogSoftmax, Module, NLLLoss, ReLU, Sequential
	from torch.nn.functional import conv2d
	from torch.nn.init import constant_, normal_, xavier_uniform_
	from torch.optim import Adamax
	from torch.utils.data import DataLoader, Dataset
	from torchvision import transforms
	from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
	from seaborn import heatmap
	from tqdm import tqdm
except Exception as e:
	print("Failed importing related libraries. Details are as follows. \n{0}\n\nPlease press the enter key to exit. ".format(e))
	if len(argv) <= 1 or "q" not in argv[1].lower():
		input()
	exit(-1)
os.chdir(os.path.abspath(os.path.dirname(__file__)))
EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EOF = (-1)
trainingCoverFolderPath = "dataSet/N"
trainingStegoFolderPath = "dataSet/YN"
validationCoverFolderPath = "dataSet/N"
validationStegoFolderPath = "dataSet/YN"
testingCoverFolderPath = "dataSet/N"
testingStegoFolderPath = "dataSet/YN"
maxEpoch = 20
trainingBatchSize = 10
validationBatchSize = 10
testingBatchSize = 20
trainingSize = [0, 3000]
validationSize = [3000, 4000]
testingSize = [4000, 5000]
initialLearningRate = 0.0001
preProcessingTuple = (0, 0, 256, 256)
checkpointFolderPath = "checkpoints"
checkpointFileNameFormatter = "YeNet_{0}.pt"
modelFilePath = "YeNetModel.pth"
resultFolderPath = "YeNetResult"
encoding = "utf-8"
dpi = 1200


class DatasetLoader(Dataset):
	def __init__(self:object, coverFolderPath:str, stegoFolderPath:str, device:object, preProcessingTuple:tuple, dataSize:int = None, transform:object = None) -> None:
		self.coverFolderPath = coverFolderPath
		self.stegoFolderPath = stegoFolderPath
		self.device = device
		self.preProcessingTuple = preProcessingTuple
		self.transforms = transform
		try:
			self.coverFilePaths = [os.path.join(self.coverFolderPath, coverName) for coverName in sorted(os.listdir(self.coverFolderPath))]
			self.coverFilePaths = [coverFilePath for coverFilePath in self.coverFilePaths if os.path.isfile(coverFilePath)]
		except Exception as e:
			print("Failed scanning the cover image folder \"{0}\". Details are as follows. \n{1}".format(self.coverFolderPath, e))
			self.coverFilePaths = []
		try:
			self.stegoFilePaths = [os.path.join(self.stegoFolderPath, stegoName) for stegoName in sorted(os.listdir(self.stegoFolderPath))]
			self.stegoFilePaths = [stegoFilePath for stegoFilePath in self.stegoFilePaths if os.path.isfile(stegoFilePath)]
		except Exception as e:
			print("Failed scanning the stego image folder \"{0}\". Details are as follows. \n{1}".format(self.stegoFolderPath, e))
			self.stegoFilePaths = []
		if isinstance(dataSize, (tuple, list)) and len(dataSize) in (2, 3):
			self.coverFilePaths = self.coverFilePaths[dataSize[0]:dataSize[1]:(dataSize[2] if len(dataSize) == 3 else 1)]
			self.stegoFilePaths = self.stegoFilePaths[dataSize[0]:dataSize[1]:(dataSize[2] if len(dataSize) == 3 else 1)]
			self.dataSize = min(len(self.coverFilePaths), len(self.stegoFilePaths))
		elif isinstance(dataSize, int):
			self.dataSize = min(dataSize, len(self.coverFilePaths), len(self.stegoFilePaths))
			indexes = sorted(randomSample(list(range(min(len(self.coverFilePaths), len(self.stegoFilePaths)))), self.dataSize))
			for i in range(len(self.coverFilePaths) - 1, -1, -1):
				if i not in indexes:
					del self.coverFilePaths[i]
			for i in range(len(self.stegoFilePaths) - 1, -1, -1):
				if i not in indexes:
					del self.stegoFilePaths[i]
		else:
			self.dataSize = min(len(self.coverFilePaths), len(self.stegoFilePaths))
			self.coverFilePaths = self.coverFilePaths[:self.dataSize]
			self.stegoFilePaths = self.stegoFilePaths[:self.dataSize]
	def resize(self:object, img:array) -> array:
		if img.ndim in (2, 3):
			if img.shape[0] == self.preProcessingTuple[2] and img.shape[1] == self.preProcessingTuple[3]:
				return img
			elif img.shape[0] > self.preProcessingTuple[2] and img.shape[1] > self.preProcessingTuple[3]:
				if self.preProcessingTuple[0] in list(range(6)):
					return array(fromarray(img).resize((self.preProcessingTuple[2], self.preProcessingTuple[3]), resample = self.preProcessingTuple[0]), dtype = "uint8")
				elif self.preProcessingTuple[0] in list(range(10, 19)):
					judgePlace = self.preProcessingTuple[0] - 10
					judgePlaceX = judgePlace % 3
					judgePlaceY = judgePlace // 3
					if judgePlaceX == 0:
						x = 0
					elif judgePlaceX == 1:
						x = (img.shape[0] - self.preProcessingTuple[2]) >> 1
					else:
						x = img.shape[0] - self.preProcessingTuple[2]
					if judgePlaceY == 0:
						y = 0
					elif judgePlaceY == 1:
						y = (img.shape[1] - self.preProcessingTuple[3]) >> 1
					else:
						y = img.shape[1] - self.preProcessingTuple[3]
					return img[x:x + self.preProcessingTuple[2], y:y + self.preProcessingTuple[3], :] if img.ndim == 3 else img[x:x + self.preProcessingTuple[2], y:y + self.preProcessingTuple[3]]
				else:
					raise ValueError("Unknown resizing method is specified. The value should be an integer selected from the following values: {0}. ".format(list(range(6)) + list(range(10, 19))))
			else:
				raise ValueError("The shape of image should not be smaller than {0} x {1}. ".format(self.preProcessingTuple[2], self.preProcessingTuple[3]))
		else:
			raise ValueError("The image should satisfy ``img.ndim in (2, 3)``. ")
	def color2grey(self:object, img:array) -> array:
		if img.ndim == 3:
			if self.preProcessingTuple[1] == 1:
				return img[:, :, 0:1]
			elif self.preProcessingTuple[1] == 2:
				return img[:, :, 1:2]
			elif self.preProcessingTuple[1] == 3:
				return img[:, :, 2:3]
			elif self.preProcessingTuple[1] == 4:
				return (npSum(img, axis = 2) // 3).astype("uint8")
			else:
				return array(fromarray(img).convert("L"), dtype = "uint8")
		elif img.ndim == 2:
			return img
		else:
			raise ValueError("The image should satisfy ``img.ndim in (2, 3)``. ")
	def preProcess(self:object, img:array) -> array:
		return self.color2grey(self.resize(img))
	def __len__(self) -> int:
		return self.dataSize
	def __getitem__(self, index:int) -> dict:
		if -self.dataSize <= index < self.dataSize:
			try:
				coverImage = self.transforms(self.preProcess(imread(self.coverFilePaths[index]))) if self.transforms else self.preProcess(imread(self.coverFilePaths[index]))
			except Exception as e:
				print("Failed reading the cover image \"{0}\". Details are as follows. \n{1}".format(self.coverFilePaths[index], e))
				return None
			try:
				stegoImage = self.transforms(self.preProcess(imread(self.stegoFilePaths[index]))) if self.transforms else self.preProcess(imread(self.stegoFilePaths[index]))
			except Exception as e:
				print("Failed reading the stego image \"{0}\". Details are as follows. \n{1}".format(self.stegoFilePaths[index], e))
				return None
			label1 = tensor(0, dtype = long).to(self.device)
			label2 = tensor(1, dtype = long).to(self.device)
			sample = {"cover":coverImage, "stego":stegoImage, "label": [label1, label2]}
			return sample
			
		else:
			return None

class ConvBlock(Module):
	def __init__(			\
		self, 			\
		in_channels:int, 		\
		out_channels:int, 		\
		kernel_size:int = 3, 		\
		stride:int = 1, 		\
		padding:int = 0, 		\
		use_pool:bool = False, 	\
		pool_size:int = 3, 		\
		pool_padding:int = 0, 	\
	) -> None:
		super().__init__()
		self.conv = Conv2d(in_channels, out_channels, kernel_size, stride = stride, padding = padding, bias = True)
		self.activation = ReLU()
		self.pool = AvgPool2d(kernel_size = pool_size, stride = 2, padding = pool_padding)
		self.use_pool = use_pool
	def forward(self, inp:tensor) -> tensor:
		if self.use_pool:
			return self.pool(self.activation(self.conv(inp)))
		return self.activation(self.conv(inp))

class SRMConv(Module):
	def __init__(self, preProcessingTuple:tuple) -> None:
		super().__init__()
		self.device = torchDevice("cuda:0" if is_available() else "cpu")
		self.srm = from_numpy(														\
			array(															\
				[														\
					[[[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]], 		\
					[[[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]], 		\
					[[[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]], 		\
					[[[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]], 		\
					[[[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]], 		\
					[[[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]], 		\
					[[[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]], 		\
					[[[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]], 		\
					[[[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, -2.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]], 		\
					[[[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, -2.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]], 		\
					[[[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, -2.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]], 		\
					[[[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]], 		\
					[[[0.0, 0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 3.0, 0.0, 0.0], [0.0, 0.0, -3.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]], 		\
					[[[0.0, 0.0, 0.0, 0.0, -1.0], [0.0, 0.0, 0.0, 3.0, 0.0], [0.0, 0.0, -3.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]], 		\
					[[[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, -3.0, 3.0, -1.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]], 		\
					[[[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -3.0, 0.0, 0.0], [0.0, 0.0, 0.0, 3.0, 0.0], [0.0, 0.0, 0.0, 0.0, -1.0]]], 		\
					[[[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, -3.0, 0.0, 0.0], [0.0, 0.0, 3.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0, 0.0]]], 		\
					[[[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, -3.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0, 0.0]]], 		\
					[[[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [-1.0, 3.0, -3.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]], 		\
					[[[-1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0, 0.0], [0.0, 0.0, -3.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]], 		\
					[[[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, -1.0, 2.0, -1.0, 0.0], [0.0, 2.0, -4.0, 2.0, 0.0], [0.0, -1.0, 2.0, -1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]], 		\
					[[[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, -1.0, 2.0, -1.0, 0.0], [0.0, 2.0, -4.0, 2.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]], 		\
					[[[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 2.0, -1.0, 0.0], [0.0, 0.0, -4.0, 2.0, 0.0], [0.0, 0.0, 2.0, -1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]], 		\
					[[[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 2.0, -4.0, 2.0, 0.0], [0.0, -1.0, 2.0, -1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]], 		\
					[[[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, -1.0, 2.0, 0.0, 0.0], [0.0, 2.0, -4.0, 0.0, 0.0], [0.0, -1.0, 2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]], 		\
					[[[-1.0, 2.0, -2.0, 2.0, -1.0], [2.0, -6.0, 8.0, -6.0, 2.0], [-2.0, 8.0, -12.0, 8.0, -2.0], [2.0, -6.0, 8.0, -6.0, 2.0], [-1.0, 2.0, -2.0, 2.0, -1.0]]], 	\
					[[[-1.0, 2.0, -2.0, 2.0, -1.0], [2.0, -6.0, 8.0, -6.0, 2.0], [-2.0, 8.0, -12.0, 8.0, -2.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]], 		\
					[[[0.0, 0.0, -2.0, 2.0, -1.0], [0.0, 0.0, 8.0, -6.0, 2.0], [0.0, 0.0, -12.0, 8.0, -2.0], [0.0, 0.0, 8.0, -6.0, 2.0], [0.0, 0.0, -2.0, 2.0, -1.0]]], 		\
					[[[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [-2.0, 8.0, -12.0, 8.0, -2.0], [2.0, -6.0, 8.0, -6.0, 2.0], [-1.0, 2.0, -2.0, 2.0, -1.0]]], 		\
					[[[-1.0, 2.0, -2.0, 0.0, 0.0], [2.0, -6.0, 8.0, 0.0, 0.0], [-2.0, 8.0, -12.0, 0.0, 0.0], [2.0, -6.0, 8.0, 0.0, 0.0], [-1.0, 2.0, -2.0, 0.0, 0.0]]]		\
				], dtype = "float32"													\
			)															\
		).to(self.device, dtype = torchFloat)
		self.tlu = Hardtanh(min_val = -3.0, max_val = 3.0)
		self.preProcessingTuple = preProcessingTuple
	def forward(self, inp:tensor) -> tensor:
		t = []
		for i in range(inp.shape[1]):
			t.append(conv2d(inp[:, i:i + 1, :, :], self.srm))
		return self.tlu(cat(t))

class YeNet(Module):
	def __init__(self, preProcessingTuple:tuple) -> None:
		super().__init__()
		self.layer1 = ConvBlock(30, 30, kernel_size = 3)
		self.layer2 = ConvBlock(30, 30, kernel_size = 3)
		self.layer3 = ConvBlock(30, 30, kernel_size = 3, use_pool = True, pool_size = 2, pool_padding = 0)
		self.layer4 = ConvBlock(30, 32, kernel_size = 5, padding = 0, use_pool = True, pool_size = 3, pool_padding = 0)
		self.layer5 = ConvBlock(32, 32, kernel_size = 5, use_pool = True, pool_padding = 0)
		self.layer6 = ConvBlock(32, 32, kernel_size = 5, use_pool = True)
		self.layer7 = ConvBlock(32, 16, kernel_size = 3)
		self.layer8 = ConvBlock(16, 16, kernel_size = 3, stride = 3)
		self.fully_connected = Sequential(Linear(in_features = 16 * 3 * 3, out_features = 2), LogSoftmax(dim = 1))
		self.preProcessingTuple = preProcessingTuple
	def forward(self, image:tensor) -> tensor:
		out = SRMConv(self.preProcessingTuple)(image)
		out = Sequential(		\
			self.layer1, 	\
			self.layer2, 	\
			self.layer3, 	\
			self.layer4, 	\
			self.layer5, 	\
			self.layer6, 	\
			self.layer7, 	\
			self.layer8		\
		)(out)
		out = out.view(out.size(0), -1)
		out = self.fully_connected(out)
		return out

class RunYeNet:
	# Load #
	def __init__(							\
		self:object, trainingCoverFolderPath:str, trainingStegoFolderPath:str, 	\
		validationCoverFolderPath:str, validationStegoFolderPath:str, 		\
		testingCoverFolderPath:str, testingStegoFolderPath:str, maxEpoch:int, 	\
		trainingBatchSize:int, validationBatchSize:int, testingBatchSize:int, 	\
		trainingSize:int, validationSize:int, testingSize:int, initialLearningRate:int, 	\
		preProcessingTuple:tuple, checkpointFolderPath:str, 			\
		checkpointFileNameFormatter:str, modelFilePath:str, 			\
		resultFilePathFormat:str, encoding:str, dpi:int			\
	) -> None:
		self.trainingCoverFolderPath = trainingCoverFolderPath
		self.trainingStegoFolderPath = trainingStegoFolderPath
		self.validationCoverFolderPath = validationCoverFolderPath
		self.validationStegoFolderPath = validationStegoFolderPath
		self.testingCoverFolderPath = testingCoverFolderPath
		self.testingStegoFolderPath = testingStegoFolderPath
		self.maxEpoch = maxEpoch
		self.trainingBatchSize = trainingBatchSize
		self.validationBatchSize = validationBatchSize
		self.testingBatchSize = testingBatchSize
		self.trainingSize = trainingSize
		self.validationSize = validationSize
		self.testingSize = testingSize
		self.initialLearningRate = initialLearningRate
		self.preProcessingTuple = preProcessingTuple
		self.checkpointFolderPath = checkpointFolderPath
		self.checkpointFileNameFormatter = checkpointFileNameFormatter
		self.modelFilePath = modelFilePath
		self.resultFolderPath = resultFolderPath
		self.encoding = encoding
		self.dpi = dpi
		self.startEpoch = 0
		self.flags = [False, False, False, False]
	def load(self:object) -> None:
		self.device = torchDevice("cuda" if is_available() else "cpu")
		trainingSet = DatasetLoader(						\
			self.trainingCoverFolderPath,  					\
			self.trainingStegoFolderPath, 					\
			self.device, 						\
			self.preProcessingTuple, 					\
			self.trainingSize, 						\
			transform = transforms.Compose(				\
				[						\
					transforms.ToPILImage(), 			\
					transforms.RandomRotation(degrees = 90), 	\
					transforms.ToTensor()			\
				]						\
			)							\
		)
		validationSet = DatasetLoader(self.validationCoverFolderPath, self.validationStegoFolderPath, self.device, self.preProcessingTuple, self.validationSize, transform = transforms.ToTensor())
		testingSet = DatasetLoader(self.validationCoverFolderPath, self.validationStegoFolderPath, self.device, self.preProcessingTuple, self.testingSize, transform = transforms.ToTensor())
		
		# Creating DataLoader objects #
		self.trainingDataLoader = DataLoader(trainingSet, batch_size = self.trainingBatchSize, shuffle = True)
		self.validationDataLoader = DataLoader(validationSet, batch_size = self.validationBatchSize, shuffle = False)
		self.testingDataLoader = DataLoader(testingSet, batch_size = self.testingBatchSize, shuffle = False)
		
		# Model creation and initialization #
		self.model = YeNet(self.preProcessingTuple)
		self.model.to(self.device)
		self.model = self.model.apply(RunYeNet.initWeight)
		self.trainingGeneralAccuracy, self.trainingDetailedAccuracy, self.trainingGeneralLoss, self.trainingDetailedLoss, self.trainingGeneralTime, self.trainingDetailedTime, self.validationGeneralAccuracy, self.validationGeneralLoss = [], [], [], [], [], [], [], []
		
		# Loss function and optimizer #
		self.lossFunction = NLLLoss()
		self.optimizer = Adamax(self.model.parameters(), lr = self.initialLearningRate, betas = (0.9, 0.999), eps = 1e-8, weight_decay = 0)
		self.flags[0] = True
	
	# Resume #
	def getLatestCheckpointEpoch(self:object) -> int:
		if os.path.isdir(self.checkpointFolderPath):
			checkpointEpochs = list(map(int, findall("\\d+", "".join(os.listdir(self.checkpointFolderPath)))))
			if checkpointEpochs:
				latestCheckpointEpoch = max(checkpointEpochs)
				if os.path.isfile(os.path.join(self.checkpointFolderPath, self.checkpointFileNameFormatter.format(latestCheckpointEpoch))):
					return latestCheckpointEpoch
				else:
					return None
			else:
				return None
		else:
			return None
	def resume(self:object) -> int:
		# Continue training from the latest checkpoint #
		latestCheckpointEpoch = self.getLatestCheckpointEpoch()
		if latestCheckpointEpoch:
			latestCheckpointFilePath = os.path.join(self.checkpointFolderPath, self.checkpointFileNameFormatter.format(latestCheckpointEpoch))
			latestCheckpoint = torchLoad(latestCheckpointFilePath)
			if "epoch" in latestCheckpoint and "model_state_dict" in latestCheckpoint and "optimizer_state_dict" in latestCheckpoint:
				self.startEpoch = latestCheckpoint["epoch"] + 1
				self.model.load_state_dict(latestCheckpoint["model_state_dict"])
				self.optimizer.load_state_dict(latestCheckpoint["optimizer_state_dict"])
				print("The latest checkpoint is loaded successfully. The training will start from Epoch {0}. ".format(latestCheckpointEpoch))
			else:
				self.startEpoch = 1
				print("The latest checkpoint is loaded but necessary items are missing. The training will start from Epoch 1. ")
		else:
			self.startEpoch = 1
			print("No checkpoints are found. The training will start from Epoch 1. ")
			if not RunYeNet.handleFolder(self.checkpointFolderPath):
				print("The checkpoint folder \"{0}\" does not exist and is not created successfully. Checkpoints may not be saved correctly. ".format(self.checkpointFolderPath))
		target = os.path.join(self.resultFolderPath, "training.txt")
		if os.path.isfile(target):
			content = RunYeNet.getTxt(target)
			if content:
				lines = [line.replace(" ", "").split("=") for line in content.split("\n")]
				for line in lines:
					if line[0] == "trainingGeneralAccuracy":
						try:
							self.trainingGeneralAccuracy = literal_eval(line[1])
						except Exception as e:
							self.trainingGeneralAccuracy = []
							print("Failed reading trainingGeneralAccuracy from \"{0}\". Details are as follows. \n{1}".format(target, e))
					elif line[0] == "trainingDetailedAccuracy":
						try:
							self.trainingDetailedAccuracy = literal_eval(line[1])
						except Exception as e:
							self.trainingDetailedAccuracy = []
							print("Failed reading trainingDetailedAccuracy from \"{0}\". Details are as follows. \n{1}".format(target, e))
					elif line[0] == "trainingGeneralLoss":
						try:
							self.trainingGeneralLoss = literal_eval(line[1])
						except Exception as e:
							self.trainingGeneralLoss = []
							print("Failed reading trainingGeneralLoss from \"{0}\". Details are as follows. \n{1}".format(target, e))
					elif line[0] == "trainingDetailedLoss":
						try:
							self.trainingDetailedLoss = literal_eval(line[1])
						except Exception as e:
							self.trainingDetailedLoss = []
							print("Failed reading trainingDetailedLoss from \"{0}\". Details are as follows. \n{1}".format(target, e))
					elif line[0] == "trainingGeneralTime":
						try:
							self.trainingGeneralTime = literal_eval(line[1])
						except Exception as e:
							self.trainingGeneralTime = []
							print("Failed reading trainingGeneralTime from \"{0}\". Details are as follows. \n{1}".format(target, e))
					elif line[0] == "trainingDetailedTime":
						try:
							self.trainingDetailedTime = literal_eval(line[1])
						except Exception as e:
							self.trainingDetailedTime = []
							print("Failed reading trainingDetailedTime from \"{0}\". Details are as follows. \n{1}".format(target, e))
					elif line[0] == "validationGeneralAccuracy":
						try:
							self.validationGeneralAccuracy = literal_eval(line[1])
						except Exception as e:
							self.validationGeneralAccuracy = []
							print("Failed reading validationGeneralAccuracy from \"{0}\". Details are as follows. \n{1}".format(target, e))
					elif line[0] == "validationGeneralLoss":
						try:
							self.validationGeneralLoss = literal_eval(line[1])
						except Exception as e:
							self.validationGeneralLoss = []
							print("Failed reading validationGeneralLoss from \"{0}\". Details are as follows. \n{1}".format(target, e))
			else:
				print("Failed reading \"{0}\". ".format(target))
		self.flags[1] = True
		return self.startEpoch
	
	# Train #
	def saveCheckpoint(self, state:dict, isPrint:bool = False) -> bool:
		target = os.path.join(self.checkpointFolderPath, self.checkpointFileNameFormatter.format(state["epoch"]))
		try:
			torchSave(state, target)
			return True
		except Exception as e:
			if isPrint:
				print("Failed saving checkpoint information to \"{0}\". Details are as follows. \n{1}".format(target, e))
			return False
	def saveModel(self:object) -> bool:
		if RunYeNet.handleFolder(os.path.split(self.modelFilePath)[0]):
			try:
				torchSave(self.model, self.modelFilePath)
				print("Save the model to \"{0}\" successfully. ".format(self.modelFilePath))
				return True
			except Exception as e:
				print("Failed saving the model to \"{0}\". Details are as follows. \n{1}".format(self.modelFilePath, e))
				return False
		else:
			print("Failed saving the model to \"{0}\" since the parent folder is not created successfully. ".format(self.modelFilePath))
			return False
	def draw(self:object, x:list, y:list, color:str = None, marker:str = None, legend:list = None, title:str = None, xlabel:str = None, ylabel:str = None, isInteger:bool = True, savefigPath:str = None, dpi:int = 1200) -> bool:
		if color and marker:
			plt.plot(x, y, color = color, marker = marker)
		elif color:
			plt.plot(x, y, color = color)
		elif marker:
			plt.plot(x, y, marker = marker)
		else:
			plt.plot(x, y)
		plt.rcParams["figure.dpi"] = 300
		plt.rcParams["savefig.dpi"] = 300
		plt.rcParams["font.family"] = "Times New Roman"
		if legend:
			plt.legend(legend)
		if title:
			plt.title(title)
		plt.gca().xaxis.set_major_locator(MaxNLocator(integer = isInteger))
		if xlabel:
			plt.xlabel(xlabel)
		if ylabel:
			plt.ylabel(ylabel)
		plt.rcParams["figure.dpi"] = dpi
		plt.rcParams["savefig.dpi"] = dpi
		if savefigPath:
			if RunYeNet.handleFolder(os.path.split(savefigPath)[0]):
				try:
					plt.savefig(savefigPath)
					plt.close()
					print("Save the figure to \"{0}\" successfully. ".format(savefigPath))
					return True
				except Exception as e:
					print("Failed saving the figure to \"{0}\". Details are as follows. \n{1}".format(savefigPath, e))
					return False
			else:
				print("Failed saving the figure to \"{0}\" since the parent folder is not created successfully. ".format(savefigPath))
				plt.show()
				plt.close()
				return False
		else:
			plt.show()
			plt.close()
			return True
	def log(self:object, logFigure:bool = False) -> bool:
		target = os.path.join(self.resultFolderPath, "training.txt")
		if RunYeNet.handleFolder(self.resultFolderPath):
			try:
				with open(target, "w", encoding = self.encoding) as f:
					f.write("trainingGeneralAccuracy = {0}\n".format(self.trainingGeneralAccuracy))
					f.write("trainingDetailedAccuracy = {0}\n".format(self.trainingDetailedAccuracy))
					f.write("trainingGeneralLoss = {0}\n".format(self.trainingGeneralLoss))
					f.write("trainingDetailedLoss = {0}\n".format(self.trainingDetailedLoss))
					f.write("trainingGeneralTime = {0}\n".format(self.trainingGeneralTime))
					f.write("trainingDetailedTime = {0}\n".format(self.trainingDetailedTime))
					f.write("validationGeneralAccuracy = {0}\n".format(self.validationGeneralAccuracy))
					f.write("validationGeneralLoss = {0}".format(self.validationGeneralLoss))
				print("Write to \"{0}\" successfully. ".format(target))
				bRet = True
			except Exception as e:
				print("Failed writing to \"{0}\". Details are as follows. \n{1}".format(target, e))
				bRet = False
			if logFigure:
				bRet = self.draw(										\
					list(range(1, len(self.trainingGeneralAccuracy) + 1)), self.trainingGeneralAccuracy, 			\
					color = "orange", marker = "x", legend = ["Accuracy"], xlabel = "Epoch", ylabel = "Accuracy", 		\
					savefigPath = os.path.join(self.resultFolderPath, "trainingGeneralAccuracy.png"), dpi = self.dpi		\
				) and bRet
				bRet = self.draw(										\
					list(range(1, len(self.trainingDetailedAccuracy) + 1)), self.trainingDetailedAccuracy, 			\
					color = "orange", marker = None, legend = ["Accuracy"], xlabel = "Batch", ylabel = "Accuracy", 	\
					savefigPath = os.path.join(self.resultFolderPath, "trainingDetailedAccuracy.png"), dpi = self.dpi		\
				) and bRet
				bRet = self.draw(									\
					list(range(1, len(self.trainingGeneralLoss) + 1)), self.trainingGeneralLoss, 			\
					color = "orange", marker = "x", legend = ["Loss"], xlabel = "Epoch", ylabel = "Loss", 		\
					savefigPath = os.path.join(self.resultFolderPath, "trainingGeneralLoss.png"), dpi = self.dpi	\
				) and bRet
				bRet = self.draw(									\
					list(range(1, len(self.trainingDetailedLoss) + 1)), self.trainingDetailedLoss, 			\
					color = "orange", marker = None, legend = ["Loss"], xlabel = "Batch", ylabel = "Loss", 	\
					savefigPath = os.path.join(self.resultFolderPath, "trainingDetailedLoss.png"), dpi = self.dpi	\
				) and bRet
				bRet = self.draw(									\
					list(range(1, len(self.trainingGeneralTime) + 1)), self.trainingGeneralTime, 			\
					color = "orange", marker = "x", legend = ["Time"], xlabel = "Epoch", ylabel = "Time (s)", 	\
					savefigPath = os.path.join(self.resultFolderPath, "trainingGeneralTime.png"), dpi = self.dpi	\
				) and bRet
				bRet = self.draw(									\
					list(range(1, len(self.trainingDetailedTime) + 1)), self.trainingDetailedTime, 		\
					color = "orange", marker = None, legend = ["Time"], xlabel = "Batch", ylabel = "Time (s)", 	\
					savefigPath = os.path.join(self.resultFolderPath, "trainingDetailedTime.png"), dpi = self.dpi	\
				) and bRet
				bRet = self.draw(										\
					list(range(1, len(self.validationGeneralAccuracy) + 1)), self.validationGeneralAccuracy, 			\
					color = "orange", marker = "x", legend = ["Accuracy"], xlabel = "Epoch", ylabel = "Accuracy", 		\
					savefigPath = os.path.join(self.resultFolderPath, "validationGeneralAccuracy.png"), dpi = self.dpi		\
				) and bRet
				bRet = self.draw(									\
					list(range(1, len(self.validationGeneralLoss) + 1)), self.validationGeneralLoss, 			\
					color = "orange", marker = "x", legend = ["Loss"], xlabel = "Epoch", ylabel = "Loss", 		\
					savefigPath = os.path.join(self.resultFolderPath, "validationGeneralLoss.png"), dpi = self.dpi	\
				) and bRet
			return bRet
		else:
			print("Failed writing to \"{0}\" since the parent folder is not created successfully. ".format(target))
			return False
	def train(self:object) -> bool:
		# Check #
		if not self.flags[0]:
			print("Please call ``load`` before ``train``. ")
			return False
		
		# Epoch #
		print("Start to train the model. ")
		try:
			for epoch in range(self.startEpoch, self.maxEpoch + 1):
				trainingAccuracy, trainingLoss, validationAccuracy, validationLoss = [], [], [], []
				epochStartTime = time()
				self.model.train()
				learningRate = self.initialLearningRate * (0.1 ** (epoch // 30))
				for paramGroup in self.optimizer.param_groups:
					paramGroup["lr"] = learningRate
				
				# Batch #
				for i, trainingBatch in enumerate(self.trainingDataLoader):
					batchStartTime = time()
					images = cat((trainingBatch["cover"], trainingBatch["stego"]), 0)
					labels = cat((trainingBatch["label"][0], trainingBatch["label"][1]), 0)
					images = images.to(self.device, dtype = torchFloat)
					labels = labels.to(self.device, dtype = long)
					self.optimizer.zero_grad()
					outputs = self.model(images)
					loss = self.lossFunction(outputs, labels)
					loss.backward()
					
					self.optimizer.step()
					trainingLoss.append(loss.item())
					prediction = outputs.data.max(1)[1]
					accuracy = prediction.eq(labels.data).sum() / (labels.size()[0])
					trainingAccuracy.append(accuracy.item())
					batchEndTime = time()
					
					self.trainingDetailedAccuracy.append(trainingAccuracy[-1])
					self.trainingDetailedLoss.append(trainingLoss[-1])
					self.trainingDetailedTime.append(batchEndTime - batchStartTime)
					
					print(													\
						"Training -> Epoch: {0} | {1}  Batch: {2} | {3}  Accuracy: {4:.4f}  Loss: {5:.4f}  LR: {6:.4f}  Time: {7:.3f}ms".format(	\
							epoch, self.maxEpoch, i + 1, len(self.trainingDataLoader), trainingAccuracy[-1], 				\
							trainingLoss[-1], self.optimizer.param_groups[0]["lr"], self.trainingDetailedTime[-1] * 1000			\
						)												\
					)
				
				# Validation #
				self.model.eval()
				with no_grad():
					for i, validationBatch in enumerate(self.validationDataLoader):
						images = cat((validationBatch["cover"], validationBatch["stego"]), 0)
						labels = cat((validationBatch["label"][0], validationBatch["label"][1]), 0)
						images = images.to(self.device, dtype = torchFloat)
						labels = labels.to(self.device, dtype = long)
						outputs = self.model(images)
						loss = self.lossFunction(outputs, labels)
						validationLoss.append(loss.item())
						prediction = outputs.data.max(1)[1]
						accuracy = prediction.eq(labels.data).sum() / (labels.size()[0])
						validationAccuracy.append(accuracy.item())
				epochEndTime = time()
				
				averageTrainingAccuracy = sum(trainingAccuracy) / len(trainingAccuracy) if trainingAccuracy else float("nan")
				averageValidationAccuracy = sum(validationAccuracy) / len(validationAccuracy) if validationAccuracy else float("nan")
				averageTrainingLoss = sum(trainingLoss) / len(trainingLoss) if trainingLoss else float("nan")
				averageValidationLoss = sum(validationLoss) / len(validationLoss) if validationLoss else float("nan")
				self.trainingGeneralAccuracy.append(averageTrainingAccuracy)
				self.trainingGeneralLoss.append(averageTrainingLoss)
				self.trainingGeneralTime.append(epochEndTime - epochStartTime)
				self.validationGeneralAccuracy.append(averageValidationAccuracy)
				self.validationGeneralLoss.append(averageValidationLoss)
				
				message = "Validation -> Epoch: {0} | {1}  Training accuracy: {2:.4f}  Validation accuracy: {3:.4f}  Training loss: {4:.4f}  Validation loss: {5:.4f}  Time: {6:.3f}s".format(	\
					epoch, self.maxEpoch, averageTrainingAccuracy, averageValidationAccuracy, averageTrainingLoss, averageValidationLoss, self.trainingGeneralTime[-1]	\
				)
				print(message)
		
				state = {							\
					"epoch":epoch, 					\
					"maxEpoch":self.maxEpoch, 				\
					"trainingBatchSize":self.trainingBatchSize, 		\
					"validationBatchSize":self.validationBatchSize, 		\
					"testingBatchSize":self.testingBatchSize, 			\
					"trainingSize":self.trainingSize, 				\
					"validationSize":self.validationSize, 			\
					"testingSize":self.testingSize, 				\
					"initialLearningRate":self.initialLearningRate, 		\
					"averageTrainingAccuracy":averageTrainingAccuracy, 	\
					"averageValidationAccuracy":averageValidationAccuracy, 	\
					"averageTrainingLoss":averageTrainingLoss, 		\
					"averageValidationLoss":averageValidationLoss, 		\
					"model_state_dict":self.model.state_dict(), 		\
					"optimizer_state_dict":self.optimizer.state_dict(), 		\
					"lr":self.optimizer.param_groups[0]["lr"]			\
				}
				self.saveCheckpoint(state)
				self.log(False)
		except KeyboardInterrupt:
			print("The training is interrupted by users. ")
		
		#  End #
		print("The training is finished. ")
		self.log(True)
		if self.saveModel():
			self.flags[2] = True
			return True
		else:
			return False
	
	# Test #
	def test(self:object) -> bool:
		# Check #
		if not self.flags[0]:
			print("Please call ``load`` before ``test``. ")
			return False
		elif not self.flags[2]:
			print("The ``train`` procedure has not been called. Trying to load the model from \"{0}\". ".format(self.modelFilePath))
			try:
				self.model = torchLoad(self.modelFilePath)
				print("Successfully load the model from \"{0}\". ".format(self.modelFilePath))
			except Exception as e:
				print("Failed loading the model from \"{0}\". Details are as follows. \n{1}".format(self.modelFilePath, e))
				return False
		
		# Test #
		print("Start to test the model. ")
		testingReal = []
		testingPredicted = []
		for i, testingBatch in enumerate(tqdm(self.testingDataLoader, ncols = 100)):
			images = cat((testingBatch["cover"], testingBatch["stego"]), 0)
			labels = cat((testingBatch["label"][0], testingBatch["label"][1]), 0)
			images = images.to(self.device, dtype = torchFloat)
			labels = labels.to(self.device, dtype = long)
			outputs = self.model(images)
			predicted = outputs.data.max(1)[1]
			testingReal += [labels[i].item() for i in range(len(labels))]
			testingPredicted += [predicted[i].item() for i in range(len(predicted))]
		
		# Output #
		try:
			testingConfusionMatrix = confusion_matrix(testingReal, testingPredicted)
			print("Testing confusion matrix: \n{0}".format(testingConfusionMatrix))
			print("Accuracy: {0}".format(accuracy_score(testingReal, testingPredicted)))
			print("Precision score: {0}".format(precision_score(testingReal, testingPredicted)))
			print("Recall score: {0}".format(recall_score(testingReal, testingPredicted)))
			print("F1 score: {0}".format(f1_score(testingReal, testingPredicted)))
			target = os.path.join(self.resultFolderPath, "testing.txt")
			if RunYeNet.handleFolder(self.resultFolderPath):
				try:
					with open(target, "w", encoding = self.encoding) as f:
						f.write("Testing confusion matrix: \n{0}\n".format(testingConfusionMatrix))
						f.write("Accuracy: {0}\n".format(accuracy_score(testingReal, testingPredicted)))
						f.write("Precision score: {0}\n".format(precision_score(testingReal, testingPredicted)))
						f.write("Recall score: {0}\n".format(recall_score(testingReal, testingPredicted)))
						f.write("F1 score: {0}".format(f1_score(testingReal, testingPredicted)))
				except Exception as e:
					print("Failed writing to \"{0}\". Details are as follows. \n{1}".format(target, e))
			else:
				print("Failed writing to \"{0}\" since the parent folder is not created successfully. ".format(target))
			plt.rcParams["figure.dpi"] = 300
			plt.rcParams["savefig.dpi"] = 300
			plt.rcParams["font.family"] = "Times New Roman"
			heatmap(testingConfusionMatrix, annot = True, fmt = "d", cmap = "BuPu")
			plt.xlabel("Predicted")
			plt.ylabel("Real")
			plt.rcParams["figure.dpi"] = self.dpi
			plt.rcParams["savefig.dpi"] = self.dpi	
			savePath = os.path.join(resultFolderPath, "testingConfusionMatrix.png")
			if RunYeNet.handleFolder(os.path.split(savePath)[0]):
				plt.savefig(savePath)
				print("Save the testing heatmap to \"{0}\" successfully. ".format(savePath))
			else:
				plt.show()
		except Exception as e:
			print("Failed generating the confusion matrix. Details are as follows. \n{0}".format(e))
		finally:
			plt.close()
		print("The testing is finished. ")
	
	# Static method #
	@staticmethod
	def initWeight(param:object) -> None:
		if isinstance(param, Conv2d):
			xavier_uniform_(param.weight.data)
			if param.bias is not None:
				constant_(param.bias.data, 0.2)
		elif isinstance(param, Linear):
			normal_(param.weight.data, mean = 0.0, std = 0.01)
			constant_(param.bias.data, 0.0)
	@staticmethod
	def getTxt(filepath, index = 0) -> str: # get .txt content
		coding = ("utf-8", "gbk", "utf-16") # codings
		if 0 <= index < len(coding): # in the range
			try:
				with open(filepath, "r", encoding = coding[index]) as f:
					content = f.read()
				return content[1:] if content.startswith("\ufeff") else content # if utf-8 with BOM, remove BOM
			except (UnicodeError, UnicodeDecodeError):
				return RunYeNet.getTxt(filepath, index + 1) # recursion
			except:
				return None
		else:
			return None # out of range
	@staticmethod
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


def main() -> int:
	try:
		runYeNet = RunYeNet(					\
			trainingCoverFolderPath, trainingStegoFolderPath, 		\
			validationCoverFolderPath, validationStegoFolderPath, 	\
			testingCoverFolderPath, testingStegoFolderPath, 		\
			maxEpoch, trainingBatchSize, validationBatchSize, 		\
			testingBatchSize, trainingSize, validationSize, testingSize, 	\
			initialLearningRate, preProcessingTuple, 			\
			checkpointFolderPath, checkpointFileNameFormatter, 	\
			modelFilePath, resultFolderPath	, encoding, dpi		\
		)
		runYeNet.load()
		runYeNet.resume()
		runYeNet.train()
		runYeNet.test()
		print("\nAll the procedures are finished. Please press the enter key to exit. \n")
		if len(argv) <= 1 or "q" not in argv[1].lower():
			input()
		return EXIT_SUCCESS
	except KeyboardInterrupt:
		print("Procedures are interrupted by users. Please press the enter key to  exit. \n")
		if len(argv) <= 1 or "q" not in argv[1].lower():
			input()
		return EXIT_FAILURE
	except Exception as e:
		print("Exceptions occurred. Details are as follows. \n{0}\n\nPlease press the enter key to exit. \n".format(e))
		if len(argv) <= 1 or "q" not in argv[1].lower():
			input()
		return EXIT_FAILURE



if __name__ == "__main__":
	exit(main())