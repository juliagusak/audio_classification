import tensorflow as tf
from glob import glob
import numpy as np
import os

import nsynth_utils


if __name__=="__main__":
    
    # source_path - path to the folder with initial signals or to the folder with their embeddinggs (one embedding ~ one .npy file)
    # saved_path - path to the folder with generated audio
    # checkpoint_path - path to the pretrained wavenet autoencoder
    # gpu_numer - gpu id
    # sample_length - signal length to synthesize
    
    # source_path = '/home/julia/DeepVoice_project/saved_data/samples/'
    # saved_path = '/home/julia/DeepVoice_project/saved_data/samples/'
    # checkpoint_path = "/home/julia/DeepVoice_data/wavenet-ckpt/model.ckpt-200000"
    
    source_path = '/workspace/data/saved_test/'
    saved_path = '/workspace/data/saved_test/'
    checkpoint_path = "/workspace/models_pretrained/wavenet-ckpt/model.ckpt-200000"
    
    postfix = ".npy"
    gpu_number = 0
    sample_length = 40000
    batch_size = 2

    if postfix == ".wav":
        files = sorted(glob('{}/**/*.wav'.format(source_path), recursive=True))
    elif postfix == ".npy":
        files = sorted(glob('{}/**/*.npy'.format(source_path), recursive=True))

    # Now synthesize from files one batch at a time
    gpu_name = "/device:GPU:{}".format(gpu_number)

    # Synthesize from files one batch at a time
    for start_file in range(0, len(files), batch_size):
        end_file = start_file + batch_size
        files_batch = files[start_file:end_file]
        save_names = [os.path.join(saved_path, 
                                   "gen_" + os.path.splitext(os.path.basename(f))[0]) + ".wav"
                      for f in files_batch]
        batch_data = nsynth_utils.load_batch(files_batch, sample_length = sample_length)

        # Encode waveforms
        encodings = batch_data if postfix == ".npy" else nsynth_utils.encode(batch_data, checkpoint_path, sample_length=sample_length)
        if gpu_number >= 0:
            with tf.device("{}".format(gpu_name)):
                nsynth_utils.synthesize(encodings, save_names, checkpoint_path) 
        else:
            nsynth_utils.synthesize(encodings, save_names, checkpoint_path)
