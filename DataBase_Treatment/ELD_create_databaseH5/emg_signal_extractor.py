from scipy import signal
import ruptures as rpt
import numpy as np
import matplotlib.pyplot as plt

def emg_tkeo(emg):
    """
    Calculates the Teager–Kaiser Energy operator.
    Args:
        emg (array): raw EMG signal.
    Returns
        tkeo (1D array_like): signal processed by the Teager–Kaiser Energy operator.
    """

    # Teager–Kaiser Energy operator
    tkeo = np.zeros(emg.shape)
    tkeo[:,1:-1] = emg[:,1:-1]*emg[:,1:-1] - emg[:,:-2]*emg[:,2:]

    return(tkeo)

def butter_lowpass_filtfilt(data, cutoff, fs, order=3):
    normal_cutoff = cutoff / (fs / 2)
    # Get the filter coefficients 
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = signal.filtfilt(b, a, data)
    
    return y

def extract_emg_signal(emg, return_rest=False, plot_result=False, Used_channels=[]):
        
        emg_data = emg.copy()
        filtered = emg_data.copy()
        
        for ch in np.arange(filtered.shape[0]):        
            
            filtered[ch,:] = np.abs(filtered[ch,:])
            filtered[ch,:] = emg_tkeo(filtered[ch,:].reshape(1, -1))
            filtered[ch,:] = filtered[ch,:] / filtered[ch,:].max()
            #filtered[ch,:] = butter_lowpass_filtfilt(filtered[ch,:], 6, 2048, order=2)
            filtered[ch,:] = butter_lowpass_filtfilt(filtered[ch,:], 3, 2048, order=2)
        
        algo = rpt.Binseg(model="l1", min_size=100, jump=20)          # l2 , l1, rbf (muito lento)
        
        if plot_result:
            y_min_f, y_max_f = filtered.min(), filtered.max()
            y_min, y_max = emg_data.min(), emg_data.max()

            # PRIMEIRO PLOT
            plt.figure(figsize=(15,10))
            for ch in range(emg_data.shape[0]):
                plt.plot(1.0 + (0.5 * filtered[ch] / y_max_f))

            for ch in range(emg_data.shape[0]):
                # plt.plot(0.5 * emg_data[ch] / y_max)
                plt.plot(emg_data[ch])
                
        if len(Used_channels) == 0:
            joined_ch = 16
            params_ch =  range(int(emg_data.shape[0] / joined_ch))
        else:
            params_ch = Used_channels
    
        result_mean = np.zeros(8)
            
        for ch in params_ch:
            if len(Used_channels) == 0:
                ch_data = filtered[joined_ch * ch:joined_ch * (ch + 1), :]
            else:
                ch_data = filtered[ch,:]
                
            result = algo.fit_predict(ch_data.T, n_bkps=7)
            
            try:
                result_mean += np.array(result)
            except:
                print(f'cant use the channel {ch}')
                
                
            if plot_result:
                plt.vlines(x=result, ymin=-1.0, ymax=1.0, color='gray', alpha=0.1)

        #result_mean = (result_mean / (emg_data.shape[0] / joined_ch))
        result_mean = (result_mean / (emg_data.shape[0] / len(Used_channels)))
        
        if plot_result:
            plt.vlines(x=result_mean, ymin=-1.0, ymax=1.0, color = 'red')
            plt.ylim([-0.7, 1.5])
            plt.show()
        
        result_mean = result_mean.astype(np.int32)    
            
        num_samples_to_consider = int(emg_data.shape[1] * 0.25)
        emg_first_part_mean = abs(emg_data[:, 0:num_samples_to_consider]).mean()
        emg_second_part_mean = abs(emg_data[:, -num_samples_to_consider:]).mean()
        
        if(emg_first_part_mean < emg_second_part_mean):
            
            if plot_result:
                if(return_rest):
                    emg_data[:, result_mean[1]:] = 0
                else:
                    emg_data[:, 0:result_mean[1]] = 0

                plt.figure(figsize=(15,3))
                for ch in range(emg_data.shape[0]):
                    plt.plot(emg_data[ch])

                plt.show()
                
            
            if(return_rest):
                emg_data = emg_data[:, 0:result_mean[1]]
            else:
                emg_data = emg_data[:, result_mean[1]:result_mean[-1]]
                
        else:
            
            if plot_result:
                if(return_rest):
                    emg_data[:, 0:result_mean[-3]] = 0
                else:
                    emg_data[:, result_mean[-3]:] = 0

                plt.figure(figsize=(15,3))
                for ch in range(emg_data.shape[0]):
                    plt.plot(emg_data[ch])

                plt.show()
                
            
            if(return_rest):
                emg_data = emg_data[:, result_mean[-3]:]
            else:
                emg_data = emg_data[:, result_mean[0]:result_mean[-3]]
        
        
        return emg_data