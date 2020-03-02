def getDetections(detectionFilePath):
    with open(detectionFilePath, 'r') as f:
        detections = [getDictFromDetection(line) for line in f]
    return detections


def getDictFromDetection(detectionStr):
    detectionList = detectionStr.split(",")
    detectionDict = {}
    detectionDict['frame'] = detectionList[0]
    detectionDict['left'] = detectionList[2]
    detectionDict['top'] = detectionList[3]
    detectionDict['width'] = detectionList[4]
    detectionDict['height'] = detectionList[5]
    detectionDict['confidence'] = detectionList[6]
    return detectionDict