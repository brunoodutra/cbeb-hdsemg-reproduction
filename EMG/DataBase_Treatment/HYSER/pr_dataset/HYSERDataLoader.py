import os
import json
import h5py
import tensorflow as tf
import numpy as np
import preprocessing as prep
from HYSERSubject import Subject
from functools import reduce
from datetime import datetime
from os.path import join


class HYSERDataLoader(tf.keras.utils.Sequence):
       
    def __init__(self, base_path, subject_id, session_id,
                       h5dir_name='train',
                       batch_size=1, 
                       shuffle=True,
                       window_size=128,
                       step_size=64,
                       preprocessing=None,
                       differential=False,
                       rest_repetition=1
                ):

        """Object Initialization Method

        Args:
            base_path (String): HYSER Database root path
            subject_id (String): Subject's ID
            session_id (int): Desired Session ID
            batch_size (int, optional): Number of EMG that should be returned. Defaults to 1.
            shuffle (bool, optional): If the database must be shuffled. Defaults to True.
            window_size (int, optional): Number of data points to be considered. Defaults to 128.
            step_size (int, optional): Number of data points passed to begin the next window. Defaults to 64.
            preprocessing (function, optional): Preprocessing functions to be applied to the EMG data. Defaults to None.
            differential (boolean, optional): EMG differential values must be calculate or not . Defaults to False.
            rest_repetition (int, optional): n-th rest sequence to save as gesture_0 when 0 is in moi. Defaults to 1 (onset).
        """
        
        self.subject_id = subject_id
        self.session_id = session_id
        self.h5dir_name = h5dir_name
        self.batchSize = batch_size
        self.shuffle = shuffle
        self.window_size = window_size
        self.step_size = step_size
        self.preprocessing = preprocessing
        self.file_paths = self.build_filepaths(base_path, subject_id, session_id)
        
        self.data = Subject(subject_name = subject_id)
        self.data.loadLabels(self.labels_dynamic_path, self.labels_maintenance_path)
        self.data.loadMultipleSamples(self.file_paths)
 
        self.differential = differential
        self.restRepetition = 1 if rest_repetition < 1 else rest_repetition


        ### Window extraction method used previously to H5 file approach
        # self.xData, self.yData = self.extract_windows(self.data, window_size, step_size)
        # self.xData = self.permute_dimensions(self.xData, (0, 2, 1))
        # print(self.xData.shape)
 
        # self.numSamples = self.xData.shape[0]
        # self.indices = np.arange(self.numSamples)
        # np.random.shuffle(self.indices)

    def get_differential(self, emg):

        emg_matrix = emg.reshape(8,8,-1)
        emg_shift = emg_matrix.copy()
        emg_shift = np.roll(emg_shift,(0,8,0))
        emg_shift[0,:,:] = np.roll(emg_shift[0,:,:],(1))

        return (emg_matrix - emg_shift).reshape(64,-1)

    def get_rest_sequence(self, sequence, rest_repetition):
        """Retrieve the (start, end) indices of the n-th True sequence.

        Same logic as HDSEMG65FileLoader.getRestSequence. Used to pick one rest
        block (typically onset) from the RMS activity mask.

        Args:
            sequence (ndarray): 1D bool array, True where the sample is rest
            rest_repetition (int): n-th rest sequence to keep (1 = first / onset)

        Returns:
            tuple[int, int]: (start, end) indices of the selected rest sequence
        """

        best_start, best_end = 0, 0
        curr_start, curr_end = 0, 0

        prev_val = sequence[0]
        repetition = 0
        find_sequence = False

        for (idx, curr_val) in enumerate(sequence[1:]):
            curr_end = idx + 1

            if prev_val == True:
                find_sequence = True
                best_start = curr_start
                best_end = curr_end
            else:
                curr_start = curr_end
                if(find_sequence):
                    repetition += 1
                    find_sequence = False
                    if(repetition == rest_repetition):
                        break

            prev_val = curr_val

        # Include the last sample when the selected rest run goes until the end of the trial.
        if repetition != rest_repetition and bool(sequence[-1]) and (repetition + 1) == rest_repetition:
            best_end = len(sequence)

        return (best_start, best_end)

    def extract_rest_emg(self, emg, active_idx, rest_repetition):
        """Slice the n-th rest block, i.e. samples not marked as active by RMS."""

        is_rest = np.ones(emg.shape[1], dtype=bool)
        if len(active_idx) > 0:
            is_rest[active_idx] = False

        idx_ini, idx_end = self.get_rest_sequence(is_rest, rest_repetition)
        return emg[:, idx_ini:idx_end]

    def get_sequences(self, emg, extract_rest=False):
        list_range = prep.removing_noise(emg, self.window_size)                            # Getting RMS-filtered EMG indexes
        filtered_emg = prep.get_emg_from_index(emg, list_range, remove_outliers=False)     # Slicing EMG signal with valid indexes only

        if not extract_rest:
            return filtered_emg

        emg_rest = self.extract_rest_emg(emg, list_range, self.restRepetition)
        return filtered_emg, emg_rest

    def save_to_h5_file(self, moi, rep):
        dir_name = self.h5dir_name
        filename = f'{dir_name}/Subject{self.subject_id}.hdf5'

        os.makedirs(dir_name, exist_ok=True) # Substituiu o if/else do mkdir

        # NOVIDADE: Verifica se o arquivo existe e se está corrompido antes de abrir
        if os.path.exists(filename):
            try:
                with h5py.File(filename, 'r'):
                    pass
            except OSError:
                print(f"Aviso: Arquivo {filename} corrompido. Removendo para recriar...")
                os.remove(filename)

        h5_file = h5py.File(filename, 'a')

        moi = list(moi)
        extract_rest = 0 in moi
        if extract_rest:
            moi = [m for m in moi if m != 0]
    
        session_gestures = []
        rep_list = []
        rest_count = 0
        session_key = 'session_' + str(self.session_id)

        if 'sessions' not in h5_file.keys():
            h5_file.create_dataset('sessions', data=np.array([]), maxshape=(None,))

        h5_file['sessions'].resize(h5_file['sessions'].shape[0] + 1, axis=0)
        h5_file['sessions'][h5_file['sessions'].shape[0]-1] = self.session_id

        if extract_rest:
            h5_file.create_group(session_key + '/gesture_0')

        for sample in self.data.samples:
            if((int(sample.getMovement()) in moi) and
               (sample.getRepetition()) in rep):

                if(int(sample.getMovement()) not in session_gestures):
                    session_gestures.append(int(sample.getMovement()))

                if(sample.getRepetition() not in rep_list):
                    rep_list.append(sample.getRepetition())

                grprep = h5_file.create_group('session_' + str(self.session_id) +
                                              '/gesture_' + sample.getMovement() + 
                                              '/repetition_' + str(sample.getRepetition()))
                
                emg_flexor = sample.getFPSignalData().T
                emg_extensor = sample.getEPSignalData().T


                if self.differential:
                    emg_flexor = self.get_differential(emg_flexor)
                    emg_extensor = self.get_differential(emg_extensor)

                if(self.preprocessing != None):
                    emg_flexor = self.preprocessing(emg_flexor)
                    emg_extensor = self.preprocessing(emg_extensor)

                emg = np.concatenate((emg_extensor, emg_flexor), axis=0)
                del emg_flexor, emg_extensor

                if extract_rest:
                    emg, emg_rest = self.get_sequences(emg, extract_rest=True)
                else:
                    emg = self.get_sequences(emg)

                grprep.create_dataset('emg', data=emg, dtype=np.float32, chunks=(128, 40))

                if extract_rest and emg_rest.shape[1] > 0:
                    n_ch, n_samp = emg_rest.shape
                    rest_grp = h5_file.create_group(session_key + '/gesture_0/repetition_' + str(rest_count))
                    rest_grp.create_dataset(
                        'emg',
                        data=emg_rest,
                        dtype=np.float32,
                        chunks=(min(128, n_ch), min(40, n_samp))
                    )
                    rest_grp.create_dataset('from_gesture', data=np.array([int(sample.getMovement())]))
                    rest_grp.create_dataset('rest_rep', data=np.array([self.restRepetition]))
                    rest_count += 1

        for gest in h5_file[session_key].keys():
            if gest == 'gesture_0':
                continue
            h5_file.create_dataset('session_'+str(self.session_id) + 
                                   '/' + gest +
                                   '/repetitions', data=rep_list)

        if extract_rest:
            h5_file.create_dataset(session_key + '/gesture_0/repetitions', data=np.arange(rest_count))
            session_gestures = [0] + session_gestures

        h5_file.create_dataset("session_" + str(self.session_id) + 
                               "/gestures", data=session_gestures)

        h5_file.close()

    @staticmethod
    def save_dataset_parameters(save_folder, moi, repetitions, subjects, sessions,
                                preprocessing=None, rest_repetition=1, differential=False,
                                window_size=128, dataset_name='HYSER'):
        """Save a json file with the parameters used to create the dataset.

        Same contract as HDSEMG65DatasetLoader.saveDatasetParameters. Training
        notebooks read ``<split>/dataset_param.json``.
        """
        os.makedirs(save_folder, exist_ok=True)

        now = datetime.now()
        dt_string = now.strftime("%Y%m%d_%H%M%S")

        gestures = list(moi)
        if 0 in gestures:
            gestures = [0] + [g for g in gestures if g != 0]

        dataset_param = {}
        dataset_param['name'] = "HYSER_" + dt_string
        dataset_param['dataset'] = {}
        dataset_param['dataset']['name'] = dataset_name
        if preprocessing is None:
            dataset_param['dataset']['preprocessing'] = []
        else:
            dataset_param['dataset']['preprocessing'] = preprocessing.getProcessingNameList()
        dataset_param['dataset']['subject'] = [f'Subject{s}' for s in subjects]
        dataset_param['dataset']['session'] = list(sessions)
        dataset_param['dataset']['gestures'] = gestures
        dataset_param['dataset']['repetition'] = list(repetitions)
        dataset_param['dataset']['rest_repetition'] = rest_repetition
        dataset_param['dataset']['differential'] = differential
        dataset_param['dataset']['window_size'] = window_size

        with open(join(save_folder, 'dataset_param.json'), 'w') as fp:
            json.dump(dataset_param, fp, indent=4)

    def getEMGbyClass(self, classes):
        emgs = np.array([])
        labels = []

        movement_file = open("hyser_movement.json", 'r')
        movement_dict = json.loads(movement_file.read())

        for i in range(len(self.xData)):
            if(movement_dict[str(self.yData[i]+1)] in classes):
                if(emgs.size == 0):
                    emgs = self.xData[i, :]
                else:
                    emgs = np.concatenate((emgs, self.xData[i, :]), axis=0)
                labels.append(self.yData[i]-1)

        return emgs.reshape(len(labels), self.window_size, emgs.shape[1]), labels

    def build_filepaths(self, base_path, subject, session):
        subject_path = os.path.join(base_path, 'subject{}_session{}'.format(subject, session))

        self.labels_dynamic_path = os.path.join(subject_path, 'label_dynamic.txt')
        self.labels_maintenance_path = os.path.join(subject_path, 'label_maintenance.txt')

        file_paths = [os.path.join(subject_path, i.split('.')[0]) 
                                                for i in os.listdir(subject_path) 
                                                if 'preprocess' in i]
        
        file_paths = list(set(file_paths))
        
        return file_paths

    def extract_windows(self, data, window_size, step_size):
        windows = []
        windows_labels = []
        
        for gesture_data in data.samples:
            emg_signal = gesture_data.getProximalData().T
            len_emg = emg_signal.shape[1]
            
            for window_ini in np.arange(0, (len_emg + 1) - window_size, step_size):

                win = emg_signal[:, window_ini:window_ini + window_size]
                windows.append(np.expand_dims(win, 0))
                windows_labels.append(int(gesture_data.getMovement()) - 1)
                
        return np.vstack(windows), np.array(windows_labels)
    
    def permute_dimensions(self, data, axis):
        return np.transpose(data, axis)
    
    def on_epoch_end(self):
        
        if self.shuffle == True:
            np.random.shuffle(self.indices)
            
    def __getitem__(self, index):

        batch_indices = self.indices[index * self.batchSize:(index + 1) * self.batchSize]
        x_data = self.xData[batch_indices, :]
        y_data = self.yData[batch_indices]

        return x_data, y_data
    
    def __len__(self):

        num_batches = self.numSamples // self.batchSize
        return num_batches
