import h5py
import numpy as np

class HDSEMG65FileLoader:
    """
    Class to extract emg data from one subject file of the 65HDSEMG dataset
    """

    def __init__(self, file_path) -> None:
        """Class constructor to initialize the object

        Args:
            file_path (str): .mat file path of the subject
        """

        self.filePath = file_path
        self.emgFile = h5py.File(file_path, 'r')

    def __del__(self):
        """Class destructor
        """

        self.emgFile.close()

    def getClassData(self):
        """Retrieve as array the class of each emg signal sample

        Returns:
            ndarray: 1D array with the class of each emg sample
        """
        
        return self.emgFile['adjusted_class'][0, :]

    def getLongestSequence(self, sequence, value):
        """Retrieve the (start, end) indices of the longest sequence of 'value' in the array 'sequence'

        Args:
            sequence (ndarray): 1D array of integers
            value (int): a integer value

        Returns:
            tuple[int, int]: (start, end) indices of the longest sequence
        """

        best_start, best_end = 0, 0
        curr_start, curr_end = 0, 0

        prev_val = sequence[0]

        for (idx, curr_val) in enumerate(sequence[1:]):
            curr_end = idx + 1 
            if prev_val == value:
                if curr_end - curr_start > best_end - best_start:
                    best_start = curr_start
                    best_end = curr_end
            else:
                curr_start = curr_end
            prev_val = curr_val

        return (best_start, best_end)

    def getRestSequence(self, sequence, rest_repetition):
        """Retrieve the (start, end) indices of one of the true sequences 

        Args:
            sequence (ndarray): 1D array with sequences of bool values
            rest_repetition (int): a integer value indicating the n-th true value sequence of interest

        Returns:
            tuple[int, int]: (start, end) indices in sequence of the intended sequence
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

        return (best_start, best_end)

    def getSelectedEMG(self, idx_start, idx_end):
        """Retrieve the emg signal and force data of the segment indicated by (idx_start, idx_end)

        Args:
            idx_start (int): start index of the segment
            idx_end (int): end index of the segment

        Returns:
            tuple[ndarray, ndarray]: (emg[ch, samples], force[ch, samples]) extracted data segment
        """

        emg_selected_extensor_data = self.emgFile['emg_extensors'][:, :, idx_start:idx_end].reshape(64, -1).astype(np.float32)
        emg_selected_flexor_data = self.emgFile['emg_flexors'][:, :, idx_start:idx_end].reshape(64, -1).astype(np.float32)
        emg_selected_data = np.concatenate((emg_selected_extensor_data, emg_selected_flexor_data), axis=0)

        force_signal = self.emgFile['force'][:, idx_start:idx_end].astype(np.float32)

        return emg_selected_data, force_signal
    
    def extractGestureEMG(self, selected_repetitions, class_index, emg_slice_class):
        """Retrieve emg data of one of the gesture repetitions performed during a period of the movement stimuli

        Args:
            selected_repetitions (list[int]): integer list of the intended n-th repetitions
            class_index (int): class of the gesture
            emg_slice_class (tuplle[int, int]): (start, end) indices indicating where is the emg signal segment with the repetitions of the gesture

        Returns:
            list[dict]: dictionary list with the data of the intended emg signals
        """

        emg_selected_data, force_signal = self.getSelectedEMG(emg_slice_class[0], emg_slice_class[1])
        emg_extracted_rep = self.emgFile['adjusted_repetition'][0, emg_slice_class[0]:emg_slice_class[1]]
        gesture_emg = []

        for repetition_ind in selected_repetitions:
            # rep_where = np.where(emg_extracted_rep == repetition_ind)[0]
            idx_ini, idx_end = self.getLongestSequence(emg_extracted_rep == repetition_ind, True)
            gesture_data = emg_selected_data[:, idx_ini:idx_end]
            gesture_force = force_signal[:, idx_ini:idx_end]
            labels = self.emgFile['adjusted_label'][:, emg_slice_class[0] + (idx_ini + idx_end) // 2] 
            gesture_emg.append({'gesture': gesture_data, 'force': gesture_force, 'class': class_index, 'labels': labels, 'raw_emg_idx': emg_slice_class, 'gesture_idx': (idx_ini, idx_end)})

        return gesture_emg
    
    def extractRestGestureEMG(self, emg_extracted_class, emg_slice_class, rest_repetition):
        """Retrieve emg data of one of the rest period performed during a period of the movement stimuli

        Args:
            emg_extracted_class (ndarray): 1D integer array with the class of the samples
            emg_slice_class (tuple[int, int]): (start, end) indices indicating where is the emg signal segment with the repetitions of the gesture
            rest_repetition (int): the n-the rest repetition of interest

        Returns:
            list[dict]: dictionary list with the data of the intended emg signals
        """

        emg_selected_data, force_signal = self.getSelectedEMG(emg_slice_class[0], emg_slice_class[1])
        gesture_emg = []

        idx_ini, idx_end = self.getRestSequence(emg_extracted_class == 0, rest_repetition)
        gesture_data = emg_selected_data[:, idx_ini:idx_end]
        gesture_force = force_signal[:, idx_ini:idx_end]
        labels = self.emgFile['adjusted_label'][:, emg_slice_class[0] + (idx_ini + idx_end) // 2] 
        gesture_emg.append({'gesture': gesture_data, 'force': gesture_force, 'class': 0, 'labels': labels, 'raw_emg_idx': emg_slice_class, 'gesture_idx': (idx_ini, idx_end)})
        
        return gesture_emg