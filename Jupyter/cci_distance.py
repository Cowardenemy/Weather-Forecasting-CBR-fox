import numpy as np


def cci_distance(windows, targetWindow, windowsLen, componentsLen, punishedSumFactor):
    print("Calculando correlación de Pearson")


    pearsonCorrelation = np.array(
        (
            [np.corrcoef(windows[currentWindow, :, currentComponent],
                         targetWindow[:, currentComponent])[
                 0][1]
             for currentWindow in range(len(windows)) for currentComponent in
             range(componentsLen)])).reshape(
        -1,
        componentsLen)
    print("Calculando distancia Euclidiana")
    euclideanDistance = np.array(
        ([np.linalg.norm(
            targetWindow[:, currentComponent] - windows[currentWindow, :, currentComponent])
            for currentWindow in range(windowsLen) for currentComponent in
            range(componentsLen)])).reshape(
        -1,
        componentsLen)
    normalizedEuclideanDistance = euclideanDistance / np.amax(euclideanDistance, axis=0)

    normalizedCorrelation = (.5 + (pearsonCorrelation - 2 * normalizedEuclideanDistance + 1) / 4)

    correlationPerWindow = np.sum(((normalizedCorrelation + punishedSumFactor) ** 2), axis=1)
    # Applying scale
    correlationPerWindow /= max(correlationPerWindow)
    return correlationPerWindow
