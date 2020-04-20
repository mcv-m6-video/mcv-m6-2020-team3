class Track(object):
    def __init__(self, id, detections):
        self.id = id
        self.detections = detections
        self.isMatched = False

    def addDetection(self, detection):
        self.detections.append(detection)

    def getDetections(self):
        return self.detections

    def setDetections(self, detections):
        self.detections = detections

    def getIsMatched(self):
        return self.isMatched

    def setIsMatched(self, isMatched):
        self.isMatched = isMatched

    def setId(self, id):
        self.id = id
