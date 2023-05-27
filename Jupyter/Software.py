import os
import pandas as pd
import numpy as np
import math
from scipy import signal
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerLine2D
from numpy import hstack
from numpy import array
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit


class cbrFox:
    def __init__(self, windows=None, target=None, targetWindow=None, num_cases=10, smoothnessFactor=.03,
                 inputNames=None,
                 outputNames=None, punishedSumFactor=.5, prediction=None):
        self.windows = windows
        self.target = target
        self.targetWindow = targetWindow
        self.num_cases = num_cases
        self.smoothnessFactor = smoothnessFactor
        self.inputNames = inputNames
        self.outputNames = outputNames
        self.punishedSumFactor = punishedSumFactor
        self.predictionTargetWindow = prediction

        self.componentsLen = self.windows.shape[2]
        self.windowLen = self.windows.shape[1]
        self.windowsLen = len(self.windows)
        self.outputComponentsLen = len(outputNames)

        # Variables used across all the code
        self.pearsonCorrelation = None
        self.euclideanDistance = None
        self.correlationPerWindow = None
        self.smoothedCorrelation = None
        self.valleyIndex = None
        self.peakIndex = None
        self.concaveSegments = None
        self.convexSegments = None
        self.bestWindowsIndex, self.worstWindowsIndex = list(), list()
        self.worstDic, self.bestDic = dict(), dict()
        self.worstSorted, self.bestSorted = dict(), dict()
        self.bestMAE, self.worstMAE = [], []
        self.analysisReport = None


    def explain(self):
        print("Calculando correlación de Pearson")
        self.pearsonCorrelation = np.array(
            ([np.corrcoef(self.windows[currentWindow, :, currentComponent], self.targetWindow[:, currentComponent])[0][1]
              for currentWindow in range(len(self.windows)) for currentComponent in range(self.componentsLen)])).reshape(-1,
                                                                                                                         self.componentsLen)
        print("Calculando distancia Euclidiana")
        self.euclideanDistance = np.array(
            ([np.linalg.norm(self.targetWindow[:, currentComponent] - self.windows[currentWindow, :, currentComponent])
              for currentWindow in range(self.windowsLen) for currentComponent in range(self.componentsLen)])).reshape(-1,
                                                                                                                       self.componentsLen)
        normalizedEuclideanDistance = self.euclideanDistance / np.amax(self.euclideanDistance, axis=0)

        normalizedCorrelation = (.5 + (self.pearsonCorrelation - 2 * normalizedEuclideanDistance + 1) / 4)

        self.correlationPerWindow = np.sum(((normalizedCorrelation + self.punishedSumFactor) ** 2), axis=1)

        self.correlationPerWindow /= max(self.correlationPerWindow)

        print("Aplicando el algoritmo LOWESS")
        self.smoothedCorrelation = lowess(self.correlationPerWindow, np.arange(len(self.correlationPerWindow)),
                                          self.smoothnessFactor)[:, 1]

        print("Extrayendo crestas y valles")
        self.valleyIndex, self.peakIndex = signal.argrelextrema(self.smoothedCorrelation, np.less)[0], \
            signal.argrelextrema(self.smoothedCorrelation, np.greater)[0]

        print("Extrayendo segmentos cóncavos y convexos")
        self.concaveSegments = np.split(np.transpose(np.array((np.arange(self.windowsLen), self.correlationPerWindow))),
                                        self.valleyIndex)

        self.convexSegments = np.split(np.transpose(np.array((np.arange(self.windowsLen), self.correlationPerWindow))),
                                       self.peakIndex)

        print("Recuperando índices originales de las ventanas")
        for split in self.concaveSegments:
            self.bestWindowsIndex.append(int(split[np.where(split == max(split[:, 1]))[0][0], 0]))
        for split in self.convexSegments:
            self.worstWindowsIndex.append(int(split[np.where(split == min(split[:, 1]))[0][0], 0]))

        self.bestDic = {index: self.correlationPerWindow[index] for index in self.bestWindowsIndex}

        self.worstDic = {index: self.correlationPerWindow[index] for index in self.worstWindowsIndex}

        self.bestSorted = sorted(self.bestDic.items(), reverse=True, key=lambda x: x[1])

        self.worstSorted = sorted(self.worstDic.items(), key=lambda x: x[1])

        self.bestSorted = self.bestSorted[0:self.num_cases]
        self.worstSorted = self.worstSorted[0:self.num_cases]

        print("Calculando MAE para cada ventana")
        for tupla in self.bestSorted:
            self.bestMAE.append(mean_absolute_error(self.target[tupla[0]], self.predictionTargetWindow.reshape(-1, 1)))

        for tupla in self.worstSorted:
            self.worstMAE.append(mean_absolute_error(self.target[tupla[0]], self.predictionTargetWindow.reshape(-1, 1)))

        print("Generando reporte de análisis")
        d = {'index': dict(self.bestSorted).keys(), 'CCI': dict(self.bestSorted).values(), "MAE": self.bestMAE,
             'index.1': dict(self.worstSorted).keys(), 'CCI.1': dict(self.worstSorted).values(), "MAE.1": self.worstMAE}
        self.analysisReport = pd.DataFrame(data=d)


    def visualizeCorrelationPerWindow(self):
        plt.figure(figsize=(17, 7))
        plt.plot(self.correlationPerWindow)
        plt.show()


    def visualizeSmoothedCorrelation(self):
        plt.figure(figsize=(17, 7))
        plt.plot(self.smoothedCorrelation)
        plt.scatter(self.peakIndex, [self.smoothedCorrelation[peak] for peak in self.peakIndex])
        plt.scatter(self.valleyIndex, [self.smoothedCorrelation[valley] for valley in self.valleyIndex])
        plt.show()

    def visualizeBestCases(self, figsize):

        fig, axs = plt.subplots(self.componentsLen, figsize=figsize)

        # COMIENZA CÓDIGO ORIGINAL
        for n_component in range(self.componentsLen):

            axs[n_component].set_title(self.inputNames[n_component])
            axs[n_component].set_ylim((0, 115))
            for i, tupla in enumerate(self.bestSorted):
                axs[n_component].plot(self.windows[tupla[0]][:, n_component], label="Caso " + str(i))

            axs[n_component].plot(self.targetWindow[:, n_component], "--", label="Caso de estudio")
            axs[n_component].legend()
        plt.show()


    def visualizeWorstCases(self, figsize):

        fig, axs = plt.subplots(self.componentsLen, figsize=figsize)

        for n_component in range(self.componentsLen):
            axs[n_component].set_title(self.inputNames[n_component])
            axs[n_component].set_ylim((0, 115))
            for i, tupla in enumerate(self.worstSorted):
                axs[n_component].plot(self.windows[tupla[0]][:, n_component], label="Caso " + str(i))
            axs[n_component].plot(self.targetWindow[:, n_component], "--", label="Caso de estudio")
            axs[n_component].legend()
        plt.show()

    def visualizeBestHistoryPredictions(self, figsize):
        fig, axs = plt.subplots(self.outputComponentsLen, figsize=figsize)

        for n_component in range(self.outputComponentsLen):

            axs[n_component].set_title(self.outputNames[n_component] + " PREDICTIONS")
            axs[n_component].set_ylim((0, 115))
            for i, tupla in enumerate(self.bestSorted):
                pass
                axs[n_component].plot(self.target[tupla[0]: tupla[0] + self.windowLen][:, n_component], label="Caso " + str(i))
                axs[n_component].scatter(self.windowLen, self.target[tupla[0] + self.windowLen + 1][n_component],
                            label="Caso " + str(i) + " predicción")
            axs[n_component].scatter(self.windowLen, self.predictionTargetWindow[:, n_component], marker="d",
                        label="Predicción del caso de estudio")
            axs[n_component].legend()
        plt.show()

    def visualizeWorstHistoryPredictions(self, figsize):

        fig, axs = plt.subplots(self.outputComponentsLen, figsize=figsize)

        for n_component in range(self.outputComponentsLen):

            axs[n_component].set_title(self.outputNames[n_component] + " PREDICTIONS")
            axs[n_component].set_ylim((0, 115))
            for i, tupla in enumerate(self.worstSorted):
                axs[n_component].plot(self.target[tupla[0]: tupla[0] + self.windowLen][:, n_component], label="Caso " + str(i))
                axs[n_component].scatter(self.windowLen, self.target[tupla[0] + self.windowLen + 1][n_component],
                                         label="Caso " + str(i) + " predicción")
            axs[n_component].scatter(self.windowLen, self.predictionTargetWindow[:, n_component], marker="d",
                                     label="Predicción del caso de estudio")
            axs[n_component].legend()
        plt.show()



    def visualizeBestCasePredictions(self):
        plt.figure(figsize=(20, 5))
        plt.ylim((0, 100))
        plt.plot(self.outputNames, self.target[-1])
        for i, tupla in enumerate(self.bestSorted):
            plt.plot(self.target[tupla[0]])
            print(mean_absolute_error(self.target[tupla[0]], self.predictionTargetWindow.reshape(-1, 1)))
        plt.plot(self.predictionTargetWindow.reshape(-1, 1), "--", label="Resultados de predicción")
        plt.show()

    def visualizeWorstCasePredictions(self):
        plt.figure(figsize=(20, 5))
        plt.ylim((0, 100))
        plt.plot(self.outputNames, self.target[-1])
        for i, tupla in enumerate(self.worstSorted):
            plt.plot(self.target[tupla[0]])
        plt.plot(self.predictionTargetWindow.reshape(-1, 1), "--", label="Resultados de predicción")
        plt.show()

    def getAnalysisreport(self):
        return self.analysisReport