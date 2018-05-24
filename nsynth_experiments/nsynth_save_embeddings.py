import numpy as np
from glob import glob
import librosa
import tensorflow as tf

from magenta.models.nsynth.wavenet.h512_bo16 import Config

import nsynth_utils

if __name__ == "__main__":
 
    source_path = '/workspace/data/LibriSpeech_to_classify/train/'
    save_path = '/workspace/data/LibriSpeech_encodings/train/'
    # source_path = '/workspace/data/test/'
    # save_path = '/workspace/data/saved_test/'
    checkpoint_path = "/workspace/models_pretrained/wavenet-ckpt/model.ckpt-200000"
    

    # sample_length - trim signal to this length and then encode
    sample_length = 40000
    batch_size = 40

    wavfiles = sorted(glob('{}/**/*.wav'.format(source_path), recursive=True))  

    # Iterate through  batches of files
    for start_file in range(0, len(wavfiles), batch_size):
        print(start_file)
        batch_number = (start_file / batch_size) + 1
        end_file = start_file + batch_size
        wavfiles_batch = wavfiles[start_file:end_file]

        # Ensure that batch of files has batch_size elements
        # Add elements to batch if needed
        batch_filter = batch_size - len(wavfiles)
        wavfiles_batch.extend(batch_filter * [wavfiles_batch[-1]])
        
        wav_data = np.array([nsynth_utils.load_audio(f, sample_length) for f in wavfiles_batch])


      # Load up the model for encoding and find the encoding of "wav_data"
        encoding = nsynth_utils.encode(wav_data, checkpoint_path, sample_length = sample_length)
    #     encoding = np.random.rand(batch_size, 16, 175)
        if encoding.ndim == 2:
            encoding = np.expand_dims(encoding, 0)
        for num, (wavfile, enc) in enumerate(zip(wavfiles_batch, encoding)):

            filename = "{}_embeddings.npy".format(wavfile.split("/")[-1].strip(".wav").strip('.flac'))
            np.save(save_path + filename, enc)

            if num + batch_filter + 1 == batch_size:
                break
