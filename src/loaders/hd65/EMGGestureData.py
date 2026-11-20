from HDSEMG65FileLoader import HDSEMG65FileLoader

class EMGGestureData:
    """
    Class to store the data of one emg gesture
    """

    def __init__(self, emg_file_path, emg_signal, class_id, labels, emg_slice_class, emg_slice_repetition, load_on_demand) -> None:
        """Class constructor to initialize the object

        Args:
            emg_file_path (str): .mat file path of the subject
            emg_signal (ndarray): 2D matrix [ch, samples] with EMG signal 
            class_id (int): a int value indicating the class of the gesture
            labels (list[int]): list of integer with the labels of the gesture
            emg_slice_class (tuple[int, int]): (start, end) indices indicating in the .mat file where is the signal segment with the repetitions of the gesture
            emg_slice_repetition (tuple[int, int]): (start, end) indices indicating in the emg_slice_class where is the gesture repetition of this object
            load_on_demand (bool): flag indicating if emg_signal have to be stored in the memory or retrieved from the file when necessary
        """

        self.emgFilePath = emg_file_path
        self.emgSignal = emg_signal
        self.classID = class_id
        self.labels = labels
        self.emgSliceClass = emg_slice_class
        self.emgSliceRepetition = emg_slice_repetition
        self.loadOnDemand = load_on_demand

        if(self.loadOnDemand):
            self.emgSignal = []

    def getEMGSignal(self):
        """Retrieve the emg signal of the gesture

        Returns:
            ndarray: 2D matrix [ch, samples] with EMG signal 
        """

        if self.loadOnDemand:
            file_loader = HDSEMG65FileLoader(self.emgFilePath)
            emg_selected_data = file_loader.getSelectedEMG(self.emgSliceClass[0] + self.emgSliceRepetition[0], self.emgSliceClass[0] + self.emgSliceRepetition[1])
            return emg_selected_data[0]
        else:
            return self.emgSignal[:, :]

    def getEMGFileLoader(self):

        """Retrieve the EMG file loader of the gesture

        Returns:
            HDSEMG65FileLoader: EMG file loader
        """
        return HDSEMG65FileLoader(self.emgFilePath)