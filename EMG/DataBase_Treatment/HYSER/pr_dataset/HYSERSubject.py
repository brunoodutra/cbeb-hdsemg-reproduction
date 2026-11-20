from HYSERSubjSample import SubjSample
import numpy as np

DYNAMIC_TASK_NAME_ROOT_LENGTH = 25
MAINTENANCE_TASK_NAME_ROOT_LENGTH = 29

class Subject():
    
    def __init__(self, samples = None, labels_dynamic = None, labels_maintenance = None, subject_name = ''):
        
        if(samples == None):
            samples = []
        if(labels_dynamic == None):
            labels_dynamic = []
        if(labels_maintenance == None):
            labels_maintenance = []

        self.samples = samples
        self.labels_dynamic = labels_dynamic
        self.labels_maintenance = labels_maintenance
        self.subject_name = subject_name
        
    def loadLabels(self, labels_dynamic_path, label_maintenance_path):
        with open(labels_dynamic_path, 'r') as labels_dynamic_file:
            self.labels_dynamic = labels_dynamic_file.read().splitlines()[0].split(',')
        with open(label_maintenance_path, 'r') as labels_maintenance_file:
            self.labels_maintenance = labels_maintenance_file.read().splitlines()[0].split(',')
    
    def loadSample(self, sample):
        self.samples.append(sample)

    def configureSample(self, task_type):
        cont = 1 if task_type == 'dynamic' else 7
        ref = ''

        labels = self.labels_dynamic if (task_type == 'dynamic') else self.labels_maintenance

        for i in range(len(labels)):
            self.getSampleByIndex(str(i+1), task_type).setMovement(labels[i])
            if(labels[i] == ref):
                self.getSampleByIndex(str(i+1), task_type).setRepetition(cont)
                cont += 1
            else:
                ref = labels[i]
                cont = 1 if task_type == 'dynamic' else 7
                
                self.getSampleByIndex(str(i+1), task_type).setRepetition(cont)
                cont += 1

    def loadMultipleSamples(self, samples_paths):

        for path in samples_paths:
            s = SubjSample()
            s.loadSample(path)
            s.setSubjectName(self.subject_name)
            self.samples.append(s)
            del s
        
        self.configureSample('dynamic')
        self.configureSample('maintenance')

    def getSampleByIndex(self, index, task_type):
        found = False
        i = 0
        
        try:
            while(not found):
                found = self.checkSampleIndex(self.samples[i], index, task_type)
                if(found):
                    return self.samples[i]
                else:
                    i += 1
        except:
            print('Index not Found.')
    
    def checkSampleIndex(self, sample, index, task_type):
        if(task_type == 'dynamic'):
            return sample.record_name[DYNAMIC_TASK_NAME_ROOT_LENGTH:].split('.')[0] == index
        elif(task_type == 'maintenance'):
            return sample.record_name[MAINTENANCE_TASK_NAME_ROOT_LENGTH:].split('.')[0] == index