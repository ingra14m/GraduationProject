import torch

def Accuracy(pred, label):
    result = pred.argmax(1)
    accuracy = (result == label).float().mean()

    return accuracy
