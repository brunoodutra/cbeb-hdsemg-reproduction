#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers, models, activations
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import h5py
from scipy.fftpack import fft
# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go


from plotly.subplots import make_subplots
class Plot_Data:

    def plot_time_series_and_fft():
        pass
    def plot_fft_subplots_dynamic(eeg_data,params,freq_ranges=[0,150],selected_channels=[0,1,6,7],fft_mode=1):
        min_freq,max_freq=freq_ranges

        fig = make_subplots(rows=len(selected_channels), cols=1, subplot_titles=[f'EEG {ch}| Channel {params["channels"][ch]}' for i,ch in enumerate(selected_channels)], 
                            vertical_spacing=0.03,  # Ajustar o espaçamento vertical entre os subplots
                            row_heights=[2.9]*len(selected_channels))  # Ajustar a altura de cada subplot

        # Plotar cada canal em um subplot
        for i, channel in enumerate(selected_channels):
            frequencies, fft_data = calculate_fft(eeg_data[:,channel],sfreq=params['sample_frequency'],mode=fft_mode)
            fig.add_trace(go.Scatter(x=frequencies, y=fft_data, mode='lines', name=f'EEG {channel}| Channel {params["channels"][i]}'), 
                        row=i+1, col=1)
            
            # Configurar intervalos dos eixos x e y para cada subplot
            fig.update_xaxes(range=[min_freq, max_freq], title_text='Frequency [Hz]', row=i+1, col=1)
            #fig.update_yaxes(range=[0, fft_data[10:].max() +0.002], title_text='Frequency Amplitude |X(t)|', row=i+1, col=1)

        # Atualizar o layout
        fig.update_layout(title='FFT', showlegend=True, height=3000, width=1000)

        fig.show()

    def plot_fft_plotly():
        pass
    def plot_psd():
        pass
    def plot_Spectrogram():
        pass
    def plot_bands(eeg_data_original,selected_channel,processing,params,T):

        filter_delta = processing.ProcessingPipeline([
            processing.BandPassFilter(.5,4, params['sample_frequency'], 3),## delta
        ]) 
        filter_theta = processing.ProcessingPipeline([
            processing.BandPassFilter(4,8, params['sample_frequency'], 3),## theta
        ]) 
        filter_alfa = processing.ProcessingPipeline([
            processing.BandPassFilter(8, 13, params['sample_frequency'], 3),## alfa
        ]) 
        filter_beta = processing.ProcessingPipeline([
            processing.BandPassFilter(13, 30, params['sample_frequency'], 3),## beta
        ]) 

        eeg_delta = filter_delta(eeg_data_original[:,params['channels'].index(selected_channel)])
        eeg_theta = filter_theta(eeg_data_original[:,params['channels'].index(selected_channel)])
        eeg_alfa = filter_alfa(eeg_data_original[:,params['channels'].index(selected_channel)])
        eeg_beta = filter_beta(eeg_data_original[:,params['channels'].index(selected_channel)])

        band_eeg_data=[eeg_delta, eeg_theta, eeg_alfa, eeg_beta]
        bands=['delta', 'theta','alfa','beta']
        # Raw Data Visualization
        fig, axs = plt.subplots(len(band_eeg_data),1, figsize=((len(band_eeg_data)*4), len(band_eeg_data)*5))

        for i, band_data in enumerate(band_eeg_data):   
            axs[i].plot(T, band_data*1000000, color='red')
            axs[i].set_title(f'Band: {bands[i]}')
            axs[i].set_xlabel('Time(s)')
            axs[i].set_ylabel('Amplitude(uV)')
            #axs[i].set_ylim(band_data[1000:-1].min(), band_data[1000:-1].max())

        fig.suptitle(f'EGG bands of channel {selected_channel}')
        #plt.tight_layout()

        plt.show()
def calculate_fft(data, sfreq=2048, resolution=4, mode=1):
    # Fast Fourier transform
    
    if mode ==1:
        # using fft with scipy.fftpack
        y_freq = fft(data)
        N = len(y_freq)
        domain = len(y_freq) // 2
        x_freq = np.linspace(0, sfreq//2, N//2)
        
        return x_freq, abs(y_freq[:domain])
    
    if mode ==2:
        # using fft with numpy
        FourierCoeff = np.fft.fft(data)/data.size
        DC = [np.abs(FourierCoeff[0])]
        amp = np.concatenate((DC, 2*np.abs(FourierCoeff[1:]))) 
        
        # compute frequencies vector until half the sampling rate
        Nyquist = sfreq/2
        #print('Nyquist frequency = %2.4f Hz'%Nyquist)
        Nsamples = int( math.floor(data.size/2) )
        hz = np.linspace(0, Nyquist, num = Nsamples + 1 )
        dhz = hz[1]
        #print('Spectral resolution = %2.4f Hz'%hz[1])

        return hz, amp[:len(hz)]
    
    if mode ==3:
        n = len(data)
        fft_result = np.fft.fft(data)
        freq = np.fft.fftfreq(n, d=1/sfreq)

        return freq[:n//2], np.abs(fft_result[:n//2])


def calculate_psd(data,sfreq=1000, scaling= 'spectrum',resolution=4, mode=1):
    
    '''
    if scaling == density obtain the power density/Hz (uV**2/Hz) spectrum
    if scaling == spectrum obtain the power (uV^2) spectrum
    '''
    if mode==1:
            (f, S)= scipy.signal.welch(data, sfreq, nperseg=None, window = 'box', scaling=scaling, nfft=int(sfreq), average='median')
    elif mode==2:
            (f, S)= scipy.signal.welch(data, fs=sfreq, nperseg=sfreq, window = 'hann', scaling=scaling, nfft=int(sfreq*resolution), average='median')
    elif mode==3:
            (f, S)= scipy.signal.welch(data, fs=sfreq, scaling=scaling, nperseg=sfreq*resolution)
    
    elif mode==4:
            
            # Perform Welch's periodogram
            segment = int( resolution*sfreq ) 
            myhann = signal.get_window('hann', segment)

            myparams1 = dict(fs = sfreq, nperseg = segment, window = myhann, noverlap = segment/2, 
                             scaling=scaling, return_onesided = True)
            (f, S) = signal.welch(x = data, **myparams1)
            S = 2*S

            dfreq = f[1]
            #print('Spectral resolution = %2.4f Hz'%dfreq)

    return f, S
def majority_voting(labels_predicted, n_voting):
    y_pred=[]
    y_model = np.zeros(n_voting)

    for pred in labels_predicted:
        y_model = np.roll(y_model,-1)
        y_model[-1] = pred

        if len(y_pred) < n_voting:
            y_pred += [pred]
        else:
            values,count = np.unique(y_model,return_counts=True)
            if len(count) == n_voting:
                y_pred += [y_pred[-1]]
            else:
                y_pred += [values[np.argmax(count)]]
    return np.array(y_pred)

def plot_history(history, model_checkpoint_path):
    fig, ax = plt.subplots(1,2, figsize=(16,8))

    ax[0].plot(history.history['loss'], color='b', label="Training loss",linewidth=3)
    ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0],linewidth=3)
    ax[0].set_title("Loss convergence ")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel('Categorical Crossentropy')
    legend = ax[0].legend(loc='best', shadow=True)

    ax[1].plot(history.history['categorical_accuracy'], color='b', label="Training accuracy",linewidth=3)
    ax[1].plot(history.history['val_categorical_accuracy'], color='r',label="Validation accuracy",linewidth=3)
    ax[1].set_title("Acurracy convergence ")
    ax[1].set_ylabel("Categorical Acurracy")
    ax[1].set_xlabel("Epochs")

    legend = ax[1].legend(loc='best', shadow=True)

    plt.savefig(model_checkpoint_path+'/Train_validatin_model.png')
    plt.show()

    # plt.savefig('Figures/'+model_checkpoint_path.split('/')[-1]+': Train_validatin_model.png')

def plot_history_accuracy(history, model_checkpoint_path):
    fig, ax = plt.subplots(1,2, figsize=(16,8))

    ax[0].plot(history.history['loss'], color='b', label="Training loss",linewidth=3)
    ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0],linewidth=3)
    ax[0].set_title("Loss convergence ")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel('Categorical Crossentropy')
    legend = ax[0].legend(loc='best', shadow=True)

    ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy",linewidth=3)
    ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy",linewidth=3)
    ax[1].set_title("Acurracy convergence ")
    ax[1].set_ylabel("Acurracy")
    ax[1].set_xlabel("Epochs")

    legend = ax[1].legend(loc='best', shadow=True)

    
    plt.savefig(model_checkpoint_path+'/Train_validation_model.png')
    # plt.show()
    # plt.savefig('Figures/'+model_checkpoint_path.split('/')[-1]+': Train_validatin_model.png')
    
def generate_data(loader, model):
    y_test = np.array([])
    y_pred = np.array([])
    for i in tqdm(range(loader.__len__())):
        xData, yData = loader.__getitem__(i)

        y_pred = np.append(y_pred, np.argmax(model.predict(xData), axis=1))
        if len(yData.shape) == 2:
            y_test = np.append(y_test, np.argmax(yData,axis=1))
        else:
            y_test = np.append(y_test, yData)
        
    return y_test, y_pred

def plot_confusion_matrix(y_test, y_pred, gesture_list, model_checkpoint_path):
    cf_matrix = confusion_matrix(y_test, y_pred)

    group_counts = ["{0:0.0f}".format(value) for value in
                    cf_matrix.flatten()]

    class_counts = np.sum(cf_matrix, axis=1)
    annotations = np.zeros((len(class_counts),len(class_counts)))

    for i in range(len(class_counts)):
        for j in range(len(class_counts)):
            annotations[i][j] = cf_matrix[i][j] / class_counts[i]

    group_percentages = ["{0:.2%}".format(value) for value in
                         annotations.flatten()]


    labels = [f"{v2}\n{v3}" for v2, v3 in
              zip(group_counts,group_percentages)]

    labels = np.asarray(labels).reshape(len(class_counts),len(class_counts))

    fig, ax = plt.subplots(figsize=(15,10))
    sns.set(font_scale=1.2) # Adjust to fit
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues', cbar=True, xticklabels=gesture_list, yticklabels=gesture_list)
    plt.title('Confusion Matrix')
    plt.savefig(model_checkpoint_path+'/Confusion_Matrix.png')
    #plt.savefig('Figures/'+str(model_checkpoint_path.split('/')[-1])+': Confusion_Matrix.png')

    ax.set_xticklabels(gesture_list, rotation = 45)
    # plt.show()
    
    
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def train_val_test_split(h5_file_name, random_state=42):
    
    with h5py.File(h5_file_name, 'r') as hf:
        y = hf['label'][...]

    x = np.arange(len(y))
    
    x_full_train, x_val, y_full_train, y_val = train_test_split(x, y, test_size=0.2, random_state=random_state)
    x_train, x_test, y_train, y_test = train_test_split(x_full_train, y_full_train, test_size=0.2, random_state=random_state)
    
    return x_train, x_val, x_test

def get_RMS(signal):
    rms_signal = np.sqrt(np.sum(signal**2)/signal.size)
    return rms_signal

def calculate_snr(signal, noise):
    # Calcula RMS do sinal
    power_signal = np.sqrt(np.sum(signal**2)/signal.size)

    # Calcula o RMS do ruído
    power_noise = np.sqrt(np.sum(noise**2)/noise.size)

    # Calcula o SNR em decibéis
    snr_db = 10 * np.log10(power_signal / power_noise)

    return snr_db

def calculate_dc_offset(signal):
    # Calcula o valor médio (DC offset) do sinal
    dc_offset = np.mean(signal)
    
    return dc_offset

def calculate_cmrr(differential_gain, common_mode_gain):
    # Calcula o CMRR em decibéis
    cmrr_db = 20 * np.log10(np.abs(differential_gain / common_mode_gain))
    
    return cmrr_db


# noise
def band_limited_noise(min_freq, max_freq, samples=1024, samplerate=1):
    freqs = np.abs(np.fft.fftfreq(samples, 1/samplerate))
    f = np.zeros(samples)
    idx = np.where(np.logical_and(freqs>=min_freq, freqs<=max_freq))[0]
    f[idx] = 1
    return fftnoise(f)

def fftnoise(f):
    f = np.array(f, dtype='complex')
    Np = (len(f) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1:Np+1] *= phases
    f[-1:-1-Np:-1] = np.conj(f[1:Np+1])
    return np.fft.ifft(f).real

def create_noise(minimum, maximum,data_len):
    x = band_limited_noise(10, 900, data_len, 2048)
    x -= np.min(x)
    x /= (np.max(x))
    x *= (maximum-minimum)
    x += minimum
    return x
    max_var=15/10

    
def RLSM_filter(Y,order=10,lambd=1):
    n_x=order
    w=np.zeros([order])
    e=np.zeros([len(Y)])
    ys=np.zeros([len(Y)])
    c=0;
    nit=len(Y)
    for k in range(order,nit):
        X=Y[k-n_x:k]
        ys[k] =  w.dot(X)                
        e[k] = Y[k] - w.dot(X)
        c = e[k] / X.dot(X)
        w= w + c * X * lambd
    return ys, w

def RLSM(Y,order,lambd=1):
    nit=len(Y)
    K=order;
    ##  Mínimos quadrados  recursivo (MQR)
    #lam = fator de esquecimento 
    P=1000*np.eye(order);
    theta=np.zeros(order);
    I=np.eye(order,order);
    phi=np.zeros(order);
    yhat=np.zeros(nit);
    W=np.zeros(nit);

    for k in range(order,nit):
        phi = Y[k-order:k];

        #yhat[k]= phi.dot(theta); 
        ek=Y[k]-phi.dot(theta);
    
        denominador= lambd + phi.dot(P).dot(phi);
        K=(1/denominador)* (P.dot(phi));
        theta= theta + K*ek;  

        phi=phi.reshape(-1,1);
        K=K.reshape(-1,1);

        # ruído do processo estocástico 
        W[k]=Y[k]-(phi.T.dot(theta));

        yhat[k]= phi.T.dot(theta); 

        P = (I - K.dot(phi.T) ).dot( P/lambd)
        P_prace=P.trace
        
    return yhat, W

# denoise ----------------------------------------------
def kalman_filter(y, H, Q, R, x0, P0):
    """
    Implementação do Filtro de Kalman.
    
    Args:
    - y: Sinal observado (com ruído).
    - H: Matriz de observação.
    - Q: Covariância do processo.
    - R: Covariância do ruído de medição.
    - x0: Estimativa inicial do estado.
    - P0: Covariância inicial do estado.

    Returns:
    - x: Estimativa do estado filtrado.
    - P: Covariância do estado filtrado.
    """
    N = len(y)
    x = np.zeros((N, len(x0)))
    P = np.zeros((N, len(x0), len(x0)))
    x[0] = x0
    P[0] = P0

    for k in range(1, N):
        # Predição do próximo estado
        x_pred = np.dot(H, x[k-1])
        P_pred = np.dot(np.dot(H, P[k-1]), H.T) + Q
        
        # Atualização utilizando a observação
        K = np.dot(np.dot(P_pred, H.T), np.linalg.inv(np.dot(np.dot(H, P_pred), H.T) + R))
        x[k] = x_pred + np.dot(K, (y[k] - np.dot(H, x_pred)))
        P[k] = P_pred - np.dot(np.dot(K, H), P_pred)

    return x, P

def adaptive_wiener_filter(d, x, M, delta):
    """
    Implementação do Filtro de Wiener Adaptativo.
    
    Args:
    - d: Sinal desejado (sinal limpo).
    - x: Sinal observado (com ruído).
    - M: Ordem do filtro.
    - delta: Parâmetro de regularização.

    Returns:
    - y: Sinal filtrado.
    """
    N = len(x)
    y = np.zeros(N)
    w = np.zeros(M)
    R = delta * np.eye(M)
    
    for n in range(N):
        # Atualização da matriz de correlação R
        x_window = x[max(0, n-M+1):n+1]
        R = delta * np.eye(M) + np.outer(x_window, x_window)
        
        # Cálculo do filtro de Wiener
        w = np.linalg.inv(R) @ x_window.T * d[n]
        
        # Estimativa do sinal desejado
        y[n] = np.dot(w, x_window)
        
    return y,w

def BMFLC(eeg_data, params, F0_list=[0, 5, 8, 11, 14], N=35, dF=0.02, MU=0.00001):
    # Output list to store reconstructed EEG signals
    eeg_reconstruct = []

    for F0 in F0_list:
        tmax = len(eeg_data) / params['sample_frequency']
        dt = 1 / params['sample_frequency']  # time step(s)
        t = np.arange(0, tmax, dt)  # time axis
        L = len(t)  # #samples

        MP = eeg_data  # Reference signal (can be modified based on the specific SSVEP experiment)

        # Additional Parameters
        v = np.zeros(N)  # array of angular frequencies
        x = np.zeros((2, N))  # reference input vector (2xN matrix) - 1st row for sin, 2nd row for cos
        w = np.zeros((2, N))  # weight vector(2xN matrix) -  1st row for sin, 2nd row for cos

        for i in range(N):
            v[i] = (F0 + dF * (i))  # assign a band of frequencies

            w[0][i] = 0
            w[1][i] = 0  # init weights

        # Initialize with zeros
        y = np.zeros(L)
        y_reconstruct = np.zeros(L)

        # BMFLC Implementation
        for k in range(N, L):
            # find reference input vector
            for i in range(N):
                x[0][i] = np.sin(2 * np.pi * v[i] * t[k])
                x[1][i] = np.cos(2 * np.pi * v[i] * t[k])

            # Find estimated signal, y
            for i in range(N):
                y[k] += w[0][i] * x[0][i] + w[1][i] * x[1][i]

            # Adapt weights
            err = MP[k] - y[k]
            for i in range(N):
                w[0][i] += 2 * MU * x[0][i] * err
                w[1][i] += 2 * MU * x[1][i] * err

            # Extract signal from estimate, y_reconstruct
            N2 = 3  # We know our frequencies are of 3 components, thus we extract the first three.
            for i in range(N2):
                y_reconstruct[k] += w[0][i] * x[0][i] + w[1][i] * x[1][i]

        eeg_reconstruct.append(y_reconstruct)
    
    return eeg_reconstruct, w


def BMFLC_LSQ(eeg_data, params, N=3, F0=8, dF=0.02):
    tmax = len(eeg_data) / params['sample_frequency']
    dt = 1 / params['sample_frequency']  # time step(s)
    t = np.arange(0, tmax, dt)  # time axis
    L = len(t)  # #samples

    v = np.zeros(N)  # array of angular frequencies
    x = np.zeros((2, N))  # reference input vector (2xN matrix) - 1st row for sin, 2nd row for cos
    w = np.zeros((2, N))  # weight vector(2xN matrix) -  1st row for sin, 2nd row for cos

    freq = F0
    for i in range(N):
        v[i] = (freq + dF * (i))  # assign a band of frequencies

    PHI = np.vstack([np.sin(2 * np.pi * v[i] * t) + np.cos(2 * np.pi * v[i] * t) for i in range(N)]).T

    y = eeg_data
    theta = np.linalg.inv(PHI.T.dot(PHI)).dot(PHI.T.dot(y))
    eeg_reconstruct = theta.dot(PHI.T)

    return eeg_reconstruct
