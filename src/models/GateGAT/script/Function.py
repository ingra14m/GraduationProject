import torch

def GetLabel(pred, label):
    result = pred.argmax(1)
    result_pred = result.cpu().data.numpy()
    result_label = label.cpu().data.numpy()

    return result_label, result_pred


def Accuracy(pred, label):
    result = pred.argmax(1)
    accuracy = (result == label).float().mean()

    return accuracy
