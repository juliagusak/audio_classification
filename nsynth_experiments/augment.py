import numpy as np
import librosa
from glob import glob



def create_noise(signal_length, noise_type = 'gaussian',
                 sigma0 = 1., sigma1 = 0.5,
                 epsilon = 0.5):
    
    if noise_type == 'gaussian':
        N = np.random.normal(scale = sigma0, size = signal_length)
    elif noise_type == 'laplacian':
        N = np.random.laplace(scale = sigma0, size = signal_length)
    elif noise_type == 'gaussianmixture':
        N = (1 - epsilon)*np.random.normal(
            scale = sigma0, size = signal_length) + epsilon*np.random.normal(
            scale = sigma1, size = signal_length)       
    else:
        print('Unknown noise type')
        N = np.zeros(signal_length)

    return N

def add_noise(signal,  noise_add = 0, noise_mult = 0):    
    return signal*(1 + noise_mult) + noise_add

   
def save_augmented(save_dir, signal_name, augmented_signals, sr = 16000):    
    for n_type, add_type, signal in augmented_signals:
        save_path = "{}/{}_{}_{}.wav".format(save_dir, signal_name, n_type, add_type)
        
        librosa.output.write_wav(save_path, signal.astype(np.float32), sr)

def augment_signal(signal):
    noised_signals = []
    for n_type in  ['gaussian', 'laplacian', 'gaussianmixture']:
        for add_type in ['add', 'addmult', 'mult']:
            if add_type != 'mult':
                sigma0 = np.random.uniform(low=0.002, high=0.01, size=1)[0]
            else:
                sigma0 = np.random.uniform(low=0.1, high=0.5, size=1)[0]
                
            if n_type != 'laplacian':
                sigma0 = np.sqrt(2)*sigma0
                
            sigma1 = 0.8*sigma0
            
            noise_add, noise_mult = 0, 0
            if add_type.startswith('add'): 
                noise_add = create_noise(len(signal), noise_type = n_type, sigma0 = sigma0, sigma1 = sigma1)
            if add_type.endswith('mult'):
                noise_mult = create_noise(len(signal), noise_type = n_type, sigma0 = sigma0, sigma1 = sigma1)  

            noised_signals.append((n_type, add_type,
                                  add_noise(signal, noise_add = noise_add, noise_mult = noise_mult)))
     
    return noised_signals


def augment_data(path, save_dir):
    files = glob("{}/**/*.wav".format(path), recursive = True)

    for wav_file in files:
        signal, sr = librosa.core.load(wav_file, sr = 16000)
        signal_name = wav_file.split('/')[-1].split('.')[0]

        augmented_signals = augment_signal(signal)
        save_augmented(save_dir, signal_name, augmented_signals, sr = sr)

        
if __name__=='__main__':
    path = "/workspace/data/LibriSpeech_to_classify/"
    save_dir = path

    for folder in ['val', 'train']:
        augment_data(path+folder, save_dir+folder)
