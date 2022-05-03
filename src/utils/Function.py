import torch

def GetLabel(pred, label):
    result = pred.argmax(1)
    result_pred = result.detach().numpy()
    result_label = label.detach().numpy()

    return result_label, result_pred


def Accuracy(pred, label):
    result = pred.argmax(1)
    accuracy = (result == label).float().mean()

    return accuracy
