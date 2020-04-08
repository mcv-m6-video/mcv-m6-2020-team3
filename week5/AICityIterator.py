import os
import glob

class AICityIterator:
    datasetStructure = {
        'S01': ['c001', 'c002', 'c003', 'c004', 'c005'],
        'S03': ['c010', 'c011', 'c012', 'c013', 'c014', 'c015'],
        'S04': ['c016', 'c018', 'c020', 'c022', 'c024', 'c026', 'c028', 'c030', 'c032', 'c034', 'c036', 'c038', 'c040','c017', 'c019', 'c021', 'c023', 'c025', 'c027', 'c029', 'c031', 'c033', 'c035', 'c037', 'c039']
    }

    def __init__(self, seq, camera, videoLength=None):
        if os.path.exists('Datasets/AIC20_track3/train/'):
            self.baseDir = 'Datasets/AIC20_track3/train/'
        elif os.path.exists('../Datasets/AIC20_track3/train/'):
            self.baseDir = '../Datasets/AIC20_track3/train/'
        else:
            raise Exception("Dataset folder not found")

        if seq not in self.datasetStructure.keys():
            raise Exception('Invalid sequence: ' + seq)

        if camera not in self.datasetStructure[seq]:
            raise Exception('Invalid camera ' + camera + ' for sequence ' + seq)

        datasetDir = self.baseDir + seq + '/' + camera + '/frames/'
        self.frameFiles = glob.glob(datasetDir + '*.jpg')
        self.frameFiles = sorted(self.frameFiles)
        self.limit = videoLength if videoLength is not None else len(self.frameFiles)
        self.currItem = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.currItem += 1
        if self.currItem < self.limit:
            return self.frameFiles[self.currItem]
        raise StopIteration

    def __len__(self):
        return len(self.frameFiles)  

def getStructure():
    datasetStructure = {
        'S01': ['c001', 'c002', 'c003', 'c004', 'c005'],
        'S03': ['c010', 'c011', 'c012', 'c013', 'c014', 'c015'],
        'S04': ['c016', 'c018', 'c020', 'c022', 'c024', 'c026', 'c028', 'c030', 'c032', 'c034', 'c036', 'c038', 'c040','c017', 'c019', 'c021', 'c023', 'c025', 'c027', 'c029', 'c031', 'c033', 'c035', 'c037', 'c039']
    }
    return datasetStructure