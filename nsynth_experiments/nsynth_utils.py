import numpy as np
import librosa
import tensorflow as tf
import os
from scipy.io import wavfile

from magenta.models.nsynth.wavenet.h512_bo16 import Config
from magenta.models.nsynth.wavenet.h512_bo16 import FastGenerationConfig



def load_audio(path, sample_length = 40000, sr = 16000):
    # audio, _ = librosa.load(path, sr = sr)
    _, audio = wavfile.read(path)
    audio = audio[:sample_length]
    return audio


def load_nsynth(batch_size = 1, sample_length = 40000):
#     Load the NSynth autoencoder network.   
    config = Config()
    print("Inside load_nsynth function")
    with tf.device("/device:GPU:0"):
        print("Loading nsynth")
        x = tf.placeholder(tf.float32, shape=[batch_size, sample_length])
        graph = config.build({"wav": x}, is_training=False)
        graph.update({"X": x})
        
    return graph

def load_fastgen_nsynth(batch_size=1):
# Load the NSynth fast generation network.
    config = FastGenerationConfig(batch_size=batch_size)
    with tf.device("/device:GPU:0"):
        x = tf.placeholder(tf.float32, shape=[batch_size, 1])
        graph = config.build({"wav": x})
        graph.update({"X": x})
    return graph


def trim_for_encoding(wav_data, sample_length, hop_length=512):
    """Make sure audio is a even multiple of hop_size.
    Args:
    wav_data: 1-D or 2-D array of floats.
    sample_length: Max length of audio data.
    hop_length: Pooling size of WaveNet autoencoder.
    Returns:
    wav_data: Trimmed array.
    sample_length: Length of trimmed array.
    """
    if wav_data.ndim == 1:
        # Max sample length is the data length
        if sample_length > wav_data.size:
            sample_length = wav_data.size
            
        # Multiple of hop_length
        sample_length = (sample_length // hop_length) * hop_length
        # Trim
        wav_data = wav_data[:sample_length]
    # Assume all examples are the same length
    elif wav_data.ndim == 2:
        # Max sample length is the data length
        if sample_length > wav_data[0].size:
            sample_length = wav_data[0].size
        # Multiple of hop_length
        sample_length = (sample_length // hop_length) * hop_length
        # Trim
        wav_data = wav_data[:, :sample_length]

    return wav_data, sample_length


def encode(wav_data, checkpoint_path, sample_length=64000):
#     Generate an array of embeddings from an array of audio.

    if wav_data.ndim == 1:
        wav_data = np.expand_dims(wav_data, 0)
        batch_size = 1
    elif wav_data.ndim == 2:
        batch_size = wav_data.shape[0]
        
    session_config = tf.ConfigProto(allow_soft_placement=True)
    session_config.gpu_options.allow_growth = True 
# настройка выше - плохая, так как ест при необходимости всю память GPU, лучше
#    session_config.gpu_options.per_process_gpu_memory_fraction = 0.4
    
    with tf.Graph().as_default(), tf.Session(config=session_config) as sess:
        hop_length = Config().ae_hop_length 
#         hop_length - это pooling size

        wav_data, sample_length = trim_for_encoding(wav_data, sample_length,
hop_length)
        net = load_nsynth(batch_size=batch_size, sample_length=sample_length)
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint_path)
        encodings = sess.run(net["encoding"], feed_dict={net["X"]: wav_data})
    return encodings



def load_batch(files, sample_length = 40000):
# """Load a batch of data from either .wav or .npy files.
# Args:
# files: A list of filepaths to .wav or .npy files
# sample_length: Maximum sample length
# Returns:
# batch_data: A padded array of audio or embeddings [batch, length, (dims)]
# """ 
    batch_data = []
    max_length = 0
    is_npy = (os.path.splitext(files[0])[1] == ".npy")
    
    # Load the data
    for f in files:
        if is_npy:
            data = np.load(f)
            batch_data.append(data)
        else:
            data = load_audio(f, sample_length, sr = 16000)
            batch_data.append(data)
        
        if data.shape[0] > max_length:
            max_length = data.shape[0]
            
    # Add padding
    for i, data in enumerate(batch_data):
        if data.shape[0] < max_length:
            if is_npy:
                padded = np.zeros([max_length, +data.shape[1]])
                padded[:data.shape[0], :] = data
            else:
                padded = np.zeros([max_length])
                padded[:data.shape[0]] = data
                
            batch_data[i] = padded
            
    # Return arrays
    batch_data = np.vstack([batch_data])
    return batch_data


                
def sample_categorical(pmf):
    """Sample from a categorical distribution.
    Args:
    pmf: Probablity mass function. Output of a softmax over categories.
    Array of shape [batch_size, number of categories]. Rows sum to 1.
    Returns:
    idxs: Array of size [batch_size, 1]. Integer of category sampled.
    """

    # почему бы не сделать это просто с помощью np.random.choice()?
    # здесь вроде делаем эквивалентные вещи: 
    # бросаем точку на отрезок [0, 1], разбитый на части длины p1,p2,..,pn
    # где pi - вероятности дискретного распределения
    
    if pmf.ndim == 1:
        pmf = np.expand_dims(pmf, 0)
    batch_size = pmf.shape[0]
    cdf = np.cumsum(pmf, axis=1)
    rand_vals = np.random.rand(batch_size)
    idxs = np.zeros([batch_size, 1])
    for i in range(batch_size):
        idxs[i] = cdf[i].searchsorted(rand_vals[i])
        # idxs[i] = np.random.choice(np.arange(len(pmf[i])), p = pmf[i])
    return idxs

def inv_mu_law_numpy(x, mu=255.0):
    """A numpy implementation of inverse Mu-Law.
    Args:
    x: The Mu-Law samples to decode.
    mu: The Mu we used to encode these samples.
    Returns:
    out: The decoded data.
    """
    x = np.array(x).astype(np.float32)
    out = (x + 0.5) * 2. / (mu + 1)
    out = np.sign(out) / mu * ((1 + mu)**np.abs(out) - 1)
    out = np.where(np.equal(x, 0), x, out)
    return out  

def save_batch(batch_audio, batch_save_paths):
    for audio, name in zip(batch_audio, batch_save_paths):
        print("Saving: {}".format(name))
        wavfile.write(name, 16000, audio)
        
def synthesize(encodings,
               save_paths,
               checkpoint_path,
               samples_per_save = 1000):
    
    hop_length = Config().ae_hop_length
    # Get lengths
    batch_size = encodings.shape[0]
    encoding_length = encodings.shape[1]
    total_length = encoding_length * hop_length
    
    session_config = tf.ConfigProto(allow_soft_placement=True)
    session_config.gpu_options.allow_growth = True
    with tf.Graph().as_default(), tf.Session(config=session_config) as sess:
        net = load_fastgen_nsynth(batch_size=batch_size)
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint_path)
        
        sess.run(net["init_ops"])
        
        audio_batch = np.zeros((batch_size, total_length), dtype = np.float32)
        audio = np.zeros([batch_size, 1])
        
        for sample_i in range(total_length):
            enc_i = sample_i//hop_length
            
            pmf = sess.run([net["predictions"], net["push_ops"]],
                           feed_dict = {net["X"]: audio,
                                        net["encoding"]: encodings[:, enc_i, :]})[0]
            sample_bin = sample_categorical(pmf)
            audio = inv_mu_law_numpy(sample_bin - 128)
            audio_batch[:, sample_i] = audio[:, 0]
            if sample_i % 100 == 0:
                print("Sample: {}".format(sample_i))
            if sample_i % samples_per_save == 0:
                save_batch(audio_batch, save_paths)
    save_batch(audio_batch, save_paths)
            
        
