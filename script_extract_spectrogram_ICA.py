import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
import pickle
import mne
import gc
import concurrent.futures
import multiprocessing

from mne.channels import make_1020_channel_selections
from mne.event import define_target_events
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


from sklearn.model_selection import KFold, train_test_split, cross_validate
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import warnings
warnings.filterwarnings("ignore")


from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
import matplotlib
matplotlib.use('Agg')

class_names = ['1','2','3']

bars = [None] * 56

os.chdir('/home/castrogaray-j/U_Winnipeg_OneDrive/Pattern recognition/BICT/Events_raw')

files = os.listdir()
pkl_files = sorted([file for file in files if file.startswith('raw') and file.endswith('.pkl')])
subjects = set([file[4:6] for file in pkl_files])

with open(pkl_files[0], 'rb') as file:
    events = pickle.load(file)

#%%capture --no-stderr --no-stdout
wind_length = 4
b_freq, u_freq = 1.0, 50.0
sfreq = events[0].info['sfreq']
freqs = np.arange(b_freq, u_freq, 1)  # Frequency range
n_cycles = freqs / 2.  # Number of cycles per frequency

ch_names = events[0].info['ch_names']

#total_iter= len(pkl_files)
#progress_bar = tqdm(total=total_iter, desc="Progress", unit="iter")

def plot_spectrogram(power, file_path):
    import matplotlib.pyplot as plt
    # Extraer datos
    times_plt = power.times
    freqs_plt = power.freqs
    data_plt = power.data[ 0, :, :]  # Seleccionar el canal correspondiente
    
    # Aplicar baseline desde el inicio hasta t=0
    baseline_idx = np.where(times_plt <= 0)[0]  # Índices donde el tiempo es <= 0
    baseline_mean = np.mean(data_plt[:, baseline_idx], axis=1, keepdims=True)  # Promedio en baseline
    
    # Aplicar corrección de baseline según 'logratio'
    data_plt = np.log10(data_plt / baseline_mean)
    
    # Configuración de límites de color
    vmin, vmax = np.percentile(data_plt, [5, 95])  # MNE ajusta entre percentiles 5 y 95 por defecto
    
    
    # Crear la figura
    fig, axes = plt.subplots(figsize=(10, 5))
    pcm = plt.pcolormesh(times_plt, freqs_plt, data_plt, shading='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)
    
    for ax in fig.axes:
                ax.set_xticks([])  # Quitar ticks en eje X
                ax.set_yticks([])  # Quitar ticks en eje Y
                ax.set_xlabel("")  # Quitar etiqueta del eje X
                ax.set_ylabel("")  # Quitar etiqueta del eje Y
                ax.set_frame_on(False)  # Quitar borde del gráfico

    plt.savefig(file_path, dpi=100, bbox_inches='tight')
    
    # Cerrar la figura para liberar memoria
    plt.close(fig)
    #del fig, axes, plt, pcm
    #gc.collect()

with open('event_labels/performance_labels', 'rb') as file:
     labels_array = pickle.load(file)

def ICA_components(raw_cleaned):
    ica_cleaned = mne.preprocessing.ICA(
        n_components=10, method="picard", max_iter="auto", random_state=97, verbose='error'
    )
    
    ica_cleaned.fit(raw_cleaned, verbose='error')

    components = ica_cleaned.get_sources(raw_cleaned)
    return components
    
def procesar_archivo(idx):
    """
    Función que procesa un archivo en paralelo
    """
    pkl = pkl_files[idx]  # Selecciona el archivo correspondiente
    labels_df = labels_array[idx]  # Obtiene los labels correspondientes
    
    with open(pkl, 'rb') as file:
        events = pickle.load(file)

    filtered_events = [item for i, item in enumerate(events[1:]) if i in labels_df.index]
    total_iter= len(filtered_events)
    bars[idx] = tqdm(total=total_iter, desc="Progress", unit="iter")
    #print("\n")
    #print(int(pkl[8:10]), int(pkl[-5]))
    for event_idx, event in enumerate(filtered_events):
        #print( event_idx, int(pkl[8:10]), int(pkl[-5]))
        #event.crop(tmin=1)
        componets = ICA_components(event)
        label = labels_df['label'].values[event_idx]
        
        ch_names = componets.info['ch_names']  # Obtener nombres de canales
        for ch in ch_names:
            power = componets.compute_tfr(method="multitaper", freqs=freqs, n_cycles=n_cycles, picks=ch)
            path = "/data/castrogaray-j/ICA_Spectrogram_images/subject_{}_class_{}_label_{}_event_{}_channel_{}.png".format(int(pkl[8:10]), int(pkl[-5]), label, event_idx, ch)
            plot_spectrogram(power, path)
        #del power
        #gc.collect()

        bars[idx].update(1)

    # Cerrar la barra de progreso al finalizar
    bars[idx].close()


if __name__ == "__main__":
    num_procesos = min(56, len(pkl_files))
    print(num_procesos)

    parametros = list(range(num_procesos))  # Lista de parámetros del 0 al 55
    
    
    
    # Crear un pool de procesos
    # with multiprocessing.Pool(processes=num_procesos) as pool:
    #     pool.map(procesar_archivo, range(num_procesos))
    with multiprocessing.Pool(processes=num_procesos) as pool:
        pool.map(procesar_archivo, range(len(pkl_files)))  # Enviar solo índices válidos