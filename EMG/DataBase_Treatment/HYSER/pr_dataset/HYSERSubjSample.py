import wfdb
import json
import numpy as np

ELECTRODE_GRID_SHAPE = {'x': 8, 'y': 8}
GRID_SIZE = 64

class SubjSample():
    def __init__(self, record_name = None, 
                       fp_signal = None, 
                       fd_signal = None, 
                       ep_signal = None, 
                       ed_signal = None, 
                       fmt = None, 
                       fs = 0, 
                       task_type = '',
                       movement = '',
                       subject_name = '',
                       session = ''):
        
        self.record_name = record_name
        self.signals = {'fp_signal': fp_signal,
                        'fd_signal': fd_signal,
                        'ep_signal': ep_signal,
                        'ed_signal': ed_signal}
        self.fmt = fmt 
        self.fs = fs
        self.task_type = task_type
        self.movement = movement
        self.subject_name = subject_name
        self.session = session
        
    def loadSample(self, sample_path):
        record = wfdb.rdrecord(sample_path).__dict__
        
        record_name = record['record_name']
        
        ed_signal = record['p_signal'][:, 0:GRID_SIZE]
        ep_signal = record['p_signal'][:, GRID_SIZE:GRID_SIZE*2]
        fd_signal = record['p_signal'][:, GRID_SIZE*2:GRID_SIZE*3]
        fp_signal = record['p_signal'][:, GRID_SIZE*3:GRID_SIZE*4]
        fmt = record['fmt']
        fs = record['fs']
        task_type = record_name.split('_')[0]
        session = sample_path.split('/')[-2][-1]
        
        self.__init__(record_name, fp_signal, fd_signal, ep_signal, ed_signal, fmt, fs, task_type, session)
    
    def plotData(self, signal_id, ax, ymin, ymax, xpos, ypos):
        try:
            ax.plot(np.arange(self.signals[signal_id].shape[0]), 
                    self.signals[signal_id][:, ((xpos - 1) * ELECTRODE_GRID_SHAPE['y'] + ypos) - 1])
#             ax.set_yticks(np.arange(ymin, ymax))
            ax.set_title('Subject {}, Movement {}, Sample {}, Sensor ({},{})'.format(self.subject_name, 
                                                                                     self.movement,
                                                                                     self.record_name,
                                                                                     xpos, ypos))
        except:
            print('Sample does not contain this type of signal')
            
    def getFPSignalData(self):
        return self.signals['fp_signal']
    
    def getFDSignalData(self):
        return self.signals['fd_signal']
    
    def getEPSignalData(self):
        return self.signals['ep_signal']
    
    def getEDSignalData(self):
        return self.signals['ed_signal']
    
    def getMovement(self):
        return self.movement
    
    def getMovementName(self):
        movement_file = open("hyser_movement.json", 'r')
        movement_dict = json.loads(movement_file.read())
        return movement_dict[self.movement]
    
    def setMovement(self, movement):
        self.movement = movement

    def getSession(self):
        return self.session
    
    def setSubjectName(self, subject_name):
        self.subject_name = subject_name

    def setRepetition(self, repetition):
        self.repetition = repetition
    
    def getRepetition(self):
        return self.repetition

    def getProximalData(self):
        return np.concatenate([self.getFPSignalData(), self.getEPSignalData()], axis=1)