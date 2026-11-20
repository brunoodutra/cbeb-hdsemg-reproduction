import os
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import numpy as np
import h5py
import itertools
from multiprocessing import Pool
from datetime import datetime
import json

from HDSEMG65FileLoader import HDSEMG65FileLoader
from EMGGestureData import EMGGestureData

class HDSEMG65DatasetLoader:
    """
    Class responsible for loading the 65HDSEMG dataset and extracting the intended gestures.
    H5 files are also created to store the intended gestures.
    """

    def __init__(self, folder_path, selected_class=list(range(0, 66)), selected_repetitions=list(range(1,6)), rest_repetition=1, subject_files=[], load_on_demand=True, save_folder="./", preprocessing=None, force_preprocessing=None) -> None:
        """Class constructor to initialize the object

        Args:
            folder_path (str): path of the folder with the emg files
            subject_files (list[str], optional): A list of strings with the name of the files (in the folder indicated by 'folder_path') to be processed. If empty, all files are processed. Defaults to [].
            selected_class (list[int], optional): Integer list with the classes of the intended gestures. Defaults to range(0, 66).
            selected_repetitions (list[int], optional): Integer list indicating the intended gesture repetitions. Defaults to range(1,6).
            rest_repetition (int, optional): Integer indicating the intended rest repetition. Defaults to 1.
            load_on_demand (bool, optional): flag indicating if extracted emg signals have to be stored in the memory or retrieved from the file when necessary. Defaults to True.
            save_folder (str, optional): path of the folder where the H5 files are saved. Defaults to "./".
            preprocessing (callable, optional): Callable object to preprocess the emg signal. Defaults to None.
            force_preprocessing (callable, optional): Callable object to preprocess the force signal. Defaults to None.
        """

        if len(subject_files) == 0:
            subject_files = listdir(folder_path)

        files = [f for f in listdir(folder_path) if (isfile(join(folder_path, f)) and f in subject_files)]

        if(rest_repetition < 1 and rest_repetition > 4):
            rest_repetition = 1

        self.flagExtractRestGesture = False
        if(0 in selected_class):
            self.flagExtractRestGesture = True
            selected_class.remove(0)

        self.subjectFiles = subject_files
        self.selectedClasses = selected_class
        self.selectedGestureRepetitions = selected_repetitions
        self.gestureDataList = [''] * len(files)
        self.loadOnDemand = load_on_demand
        self.restRepetition = rest_repetition
        self.preprocessing = preprocessing
        self.forcePreprocessing = force_preprocessing
        self.folderPath = folder_path
        self.saveFolder = save_folder
        
        os.makedirs(self.saveFolder, exist_ok=True) # make the path folder
       
        with Pool() as pool:
        # call the same function with different data in parallel
            self.gestureDataList = pool.map(self.processGestureFile, files)

        self.gestureDataList = list(itertools.chain.from_iterable(self.gestureDataList))

        self.saveDatasetParameters()

    def processGestureFile(self, f):
        """Process a subject file for extracting the intended gestures

        Args:
            f (str): file path

        Returns:
            list[EMGGestureData]: object list with the extracted gestures
        """

        emg_file_loader = HDSEMG65FileLoader(join(self.folderPath, f))                          # Loading subject's data

        h5_file_path = f.replace('.mat', '')
        h5_file = h5py.File(join(self.saveFolder, h5_file_path + '.hdf5'), 'w')

        h5_file.create_dataset("sessions", data=[0])
        h5_file.create_group("session_0")

        if(self.flagExtractRestGesture):
            h5_file.create_dataset("session_0/gestures", data=np.array([0] + self.selectedClasses))
        else:
            h5_file.create_dataset("session_0/gestures", data= np.array(self.selectedClasses))

        return self.getGesturesFromFile(emg_file_loader, h5_file)

    def extractGestures(self, emg_file_loader, emg_slice_class, emg_class, class_index, h5_file):
        """_summary_

        Args:
            emg_file_loader (HDSEMG65FileLoader): emg file loader where the data will be read from
            emg_slice_class (tuplle[int, int]): (start, end) indices indicating where is the emg signal segment with the repetitions of the gesture
            emg_class (ndarray): 1D integer array with the class of the samples
            class_index (int): class of the gesture
            h5_file (h5py): H5 object where the data will be saved to

        Returns:
            list[EMGGestureData]: object list with the extracted gestures
        """

        gesture_data_list = []
        extracted_gestures = emg_file_loader.extractGestureEMG(self.selectedGestureRepetitions, class_index, emg_slice_class)

        for i, gest in enumerate(extracted_gestures):
            if(i == 0):
                h5_file.create_dataset("session_0/gesture_" + str(class_index) + "/labels", data=gest['labels'])

            h5_file_group_rep = h5_file.create_group("session_0/gesture_" + str(class_index) + "/repetition_" + str(self.selectedGestureRepetitions[i]))
            
            if(self.preprocessing != None):
                h5_file_group_rep.create_dataset("emg", data=self.preprocessing(gest['gesture']), dtype='float32', chunks=(128, 40))
            else:
                h5_file_group_rep.create_dataset("emg", data=gest['gesture'], dtype='float32', chunks=(128, 40))

            if(self.forcePreprocessing != None):
                h5_file_group_rep.create_dataset("force", data=self.forcePreprocessing(gest['force']), dtype='float32', chunks=(9, 40))
            else:
                h5_file_group_rep.create_dataset("force", data=gest['force'], dtype='float32', chunks=(9, 40))

        if(self.flagExtractRestGesture):    
            extracted_rest_gestures = emg_file_loader.extractRestGestureEMG(emg_class[emg_slice_class[0]: emg_slice_class[1]], emg_slice_class, self.restRepetition)

            rest_count = h5_file['session_0']['gesture_0']['repetitions'][0]
            for i, gest in enumerate(extracted_rest_gestures):
                h5_file_group_rep = h5_file.create_group("session_0/gesture_0/repetition_" + str(rest_count))

                if(self.preprocessing != None):
                    h5_file_group_rep.create_dataset("emg", data=self.preprocessing(gest['gesture']), dtype='float32', chunks=(128, 40))
                else:
                    h5_file_group_rep.create_dataset("emg", data=gest['gesture'], dtype='float32', chunks=(128, 40))

                if(self.forcePreprocessing != None):
                    h5_file_group_rep.create_dataset("force", data=self.forcePreprocessing(gest['force']), dtype='float32', chunks=(9, 40))
                else:
                    h5_file_group_rep.create_dataset("force", data=gest['force'], dtype='float32', chunks=(9, 40))
                
                h5_file_group_rep.create_dataset("rest_rep", data=np.array([self.restRepetition]))
                h5_file_group_rep.create_dataset("from_gesture", data=np.array([class_index]))
                rest_count += 1

            h5_file['session_0']['gesture_0']['repetitions'][0] = rest_count

            extracted_gestures += extracted_rest_gestures

        for gest in extracted_gestures:
            gesture_data_list.append(EMGGestureData(emg_file_loader.filePath, gest['gesture'], gest['class'], gest['labels'], gest['raw_emg_idx'], gest['gesture_idx'], self.loadOnDemand))

        return gesture_data_list

    def getGesturesFromFile(self, emg_file_loader, h5_file):
        """Extract the intended gestures of the emg file

        Args:
            emg_file_loader (HDSEMG65FileLoader): emg file loader where the data will be read from
            h5_file (h5py): H5 object where the data will be saved to

        Returns:
            list[EMGGestureData]: object list with the extracted gestures
        """

        emg_class = emg_file_loader.getClassData()
        gesture_data_list = []

        if(self.flagExtractRestGesture):
            h5_file.create_group("session_0/gesture_0")
            h5_file.create_dataset("session_0/gesture_0/labels", data=[0] * 16)
            h5_file.create_dataset("session_0/gesture_0/repetitions", data=[0])

        for class_index in self.selectedClasses:
            data_ind = np.where(emg_class == class_index)[0]
            
            if(class_index == 58):
                last_index = 0
                for i in range(data_ind.min(), data_ind.max()):
                    if(emg_class[i] == 59):
                        h5_file.create_group("session_0/gesture_" + str(class_index))
                        h5_file.create_dataset("session_0/gesture_" + str(class_index) + "/repetitions", data=self.selectedGestureRepetitions)
                        gesture_data_list += self.extractGestures(emg_file_loader, (data_ind.min(), i), emg_class, class_index, h5_file)
                        last_index = i
                        break
                
                # for i in range(last_index, data_ind.max()):
                #     if(emg_class[i] == 58):
                #         gesture_data_list += self.extractGestures(emg_file_loader, (i, data_ind.max()), emg_class, class_index, h5_file)
                #         break
            else:
                h5_file.create_group("session_0/gesture_" + str(class_index))
                h5_file.create_dataset("session_0/gesture_" + str(class_index) + "/repetitions", data=self.selectedGestureRepetitions)
                gesture_data_list += self.extractGestures(emg_file_loader, (data_ind.min(), data_ind.max()), emg_class, class_index, h5_file)

        if(self.flagExtractRestGesture):
            rest_count = h5_file['session_0']['gesture_0']['repetitions'][0]
            del h5_file['session_0']['gesture_0']['repetitions']
            h5_file.create_dataset("session_0/gesture_0/repetitions", data=np.arange(rest_count))

        return gesture_data_list

    def plotData(self, channel=0):

        """
        Plot the emg signal of the extracted
        """

        current_class = -1
        for gesture_data in self.gestureDataList:

            if(current_class != gesture_data.classID):
                current_class = gesture_data.classID
                file_loader = gesture_data.getEMGFileLoader()
                emg_selected_data = file_loader.getSelectedEMG(gesture_data.emgSliceClass[0], gesture_data.emgSliceClass[1])[0]
                emg_extracted_rep = file_loader.emgFile['adjusted_repetition'][0, gesture_data.emgSliceClass[0]:gesture_data.emgSliceClass[1]]
                y_range = emg_selected_data[channel,:].min(), emg_selected_data[channel,:].max() 

                plt.figure(figsize=(15,5))
                plt.ylim(y_range)
                plt.plot(emg_selected_data[channel,:])
                plt.plot(emg_extracted_rep)
                plt.title("emg class " + str(gesture_data.classID) + " - " + file_loader.emgFile.filename)

            aux = np.zeros(emg_selected_data.shape[1])
            aux[gesture_data.emgSliceRepetition[0]:gesture_data.emgSliceRepetition[1]] = gesture_data.getEMGSignal()[channel, :]

            plt.figure(figsize=(15,5))
            plt.ylim(y_range)
            plt.plot(aux)
            plt.title("rep class" + str(gesture_data.classID))
    
    def getGestureData(self):
        """Retrieve the list of extracted gestures

        Returns:
            list[EMGGestureData]: object list with the extracted gestures
        """
        return self.gestureDataList

    def saveDatasetParameters(self):
        """Save a json file with the parameters used to create the dataset
        """
        now = datetime.now()
        dt_string = now.strftime("%Y%m%d_%H%M%S")

        dataset_param = {}
        dataset_param['name'] = "65HDsEMG_" + dt_string
        dataset_param['dataset'] = {}
        dataset_param['dataset']['name'] = '65HDsEMG'
        
        if self.preprocessing is None:
            dataset_param['dataset']['preprocessing'] = []
        else:
            dataset_param['dataset']['preprocessing'] = self.preprocessing.getProcessingNameList()
        
        if self.forcePreprocessing is None:
            dataset_param['dataset']['force_preprocessing'] = []
        else:
            dataset_param['dataset']['force_preprocessing'] = self.forcePreprocessing.getProcessingNameList()
            
        dataset_param['dataset']['subject'] = self.subjectFiles
        dataset_param['dataset']['session'] = [0]
        dataset_param['dataset']['gestures'] = self.selectedClasses
        dataset_param['dataset']['repetition'] = self.selectedGestureRepetitions
        dataset_param['dataset']['rest_repetition'] = self.restRepetition

        if(self.flagExtractRestGesture):
            dataset_param['dataset']['gestures'] = [0] + self.selectedClasses

        with open(join(self.saveFolder, 'dataset_param.json'), 'w') as fp:
            json.dump(dataset_param, fp, indent=4)