import glob
import pandas as pd
import os, sys
import h5py
import numpy as np
from emg_signal_extractor import *

path = os.path.abspath('./../../Processing/')
if(path not in sys.path):
    sys.path.append(path)
import ProcessingPipeline as pp

filter_processing = pp.ProcessingPipeline([
    pp.HighPassFilter(10, 2034.179688, 3),## lee_2019 reference papaer
    pp.LowPassFilter(450, 2034.179688, 3),## lee_2019 reference papaer
    #pp.NotchFilter(60,60.0,params['sampling_frequency'])
])

def extract_gesture_description(file_name):
    
    gesture_description = file_name.split('_')[0]
    # print(gesture_description)

    gesture_description = gesture_description.split('-')
    # print(gesture_description)

    session = int(gesture_description[0][1:])
    gesture_class = int(gesture_description[1][1:])
    repetition = int(gesture_description[2][1:])
    gesture_type = gesture_description[3]
    
    return session, gesture_class, repetition, gesture_type

def convert_csv_to_h5_emg_file(subject_folder_path, extract_emg=False, Used_channels=[]):
    
    files_list = glob.glob(subject_folder_path + '*.csv', recursive=False)
    
    session_rest_count = {}
    session_gestures = {}
    sessions = []
    
    h5_file = h5py.File(subject_folder_path + 'emg_file.hdf5', 'w')
    
    for file_path in files_list:
        
        print('processing file', file_path, '...')

        data = pd.read_csv(file_path)
        emg_data = data.to_numpy().T
        channel_names = list(data.columns.values)
        
        #emg_data=filter_processing(emg_data)
        
        file_name = os.path.split(file_path)[1]    
        session_id, gesture_class, repetition, gesture_type = extract_gesture_description(file_name)

        if session_id not in sessions:
            sessions.append(session_id)
            session_rest_count[session_id] = 1
            session_gestures[session_id] = []

        if gesture_type == 'G':
            grprep = h5_file.create_group('session_' + str(session_id) +
                                          '/gesture_' + str(gesture_class) + 
                                          '/repetition_' + str(repetition))

            if gesture_class not in session_gestures[session_id]:
                session_gestures[session_id].append(gesture_class)
                
            if extract_emg:
                emg_data = extract_emg_signal(emg_data, return_rest=False, plot_result=False, Used_channels= Used_channels)

        elif gesture_type == 'R':
            grprep = h5_file.create_group('session_' + str(session_id) +
                                          '/gesture_0' + 
                                          '/repetition_' + str(session_rest_count[session_id]))

            if 0 not in session_gestures[session_id]:
                session_gestures[session_id].append(0)
                
            if extract_emg:
                emg_data = extract_emg_signal(emg_data, return_rest=True, plot_result=False, Used_channels= Used_channels)

            grprep.create_dataset('from_gesture', data=np.array([gesture_class]))

            session_rest_count[session_id] += 1

        grprep.create_dataset('emg', data=emg_data, dtype=np.float32, chunks=(64, 32))

    h5_file.create_dataset('sessions', data=np.array(sessions))

    for s in sessions:
        h5_file.create_dataset('session_' + str(s) + '/gestures', data=np.array(session_gestures[s]))
        
    h5_file.close()
    
def create_data(source, destination, session_list, repetitions_list, gestures_list, processing=None):
        database_path = os.path.dirname(destination)
        if not os.path.exists(database_path):
            os.makedirs(database_path)
            print(f'Creating the {database_path}.')
        else:
            print(f'Using the {database_path} path.')

        with h5py.File(source, 'r') as h5_file:
           
            
            with h5py.File(destination, 'w') as h5_processed_file:
                # Iterate over sessions in the original file
                
                h5_processed_file.create_dataset('sessions', data= np.array(session_list))
                
                h5_processed_file.create_group(f'session_{s}')
                for s in session_list:
                    # Create a group for the new session in the processed file
                    session_group = h5_processed_file.create_group(f'session_{s}')

                    # Iterate over gestures in the current session
                    for g in gestures_list:
                        # Create a group for the new gesture in the processed file
                        gesture_group = session_group.create_group(f'gesture_{g}')

                        # Iterate over repetitions in the current gesture
                        for r in repetitions_list:
                            # Apply preprocessing steps to the data
                            if processing != None:
                                emg_dabase = processing(h5_file[f'session_{s}'][f'gesture_{g}'][f'repetition_{r}']['emg'][:])
                            else:
                                emg_dabase = h5_file[f'session_{s}'][f'gesture_{g}'][f'repetition_{r}']['emg'][:]
                            # Create a group for the new repetition in the processed file
                            repetition_group = gesture_group.create_group(f'repetition_{r}')

                            # Save the preprocessed data in the new group
                            repetition_group.create_dataset('emg', data=emg_dabase)