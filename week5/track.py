class Track(object):
    def __init__(self, id, detections):
        self.id = id
        self.detections = detections

    def addDetection(self, detection):
        self.detections.append(detection)


