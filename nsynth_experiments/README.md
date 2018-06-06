# Experiments
This directory contains some experiments with WaveNet-based autoencoder
([arXiv paper](https://arxiv.org/abs/1704.01279), [blog post](https://magenta.tensorflow.org/nsynth),
[GitHub](https://github.com/tensorflow/magenta/tree/master/magenta/models/nsynth)) also known as NSynth model.

# The Models
We were inspired by NSynth model proposed by Google Magenta team (test 
[this notebook](https://colab.research.google.com/notebooks/magenta/nsynth/nsynth.ipynb) to play with the model)
and decided to use it in audio classification tasks (speaker identification, environment sound classification). 

  * #### Finetuned encoder
  
  We took an encoder part of Nsynth model and add classification neural network on top of it. 
  We initialized the encoder part with weights from pretrained NSynth model.  We train
  the whole neural network to minimize cross entropy score. 
  
  * #### Finetuned autoencoder
  
  We took the whole autoencoder from Nsynth model
  and add classification neural network on top of its encoder
  (so, we obtained a neural network with two heads: first one with synthezed audio as output,
  second one with probabilities of class affiliation as output).
  We initialized the autoencoder with weights from pretrained NSynth model. 
  We train  the whole neural network to minimize sum of two losses
  (one loss corresponds to one head). 
  
# Training
To train the models you need 
  * #### Dataset containing raw audio 
  We train our models on [LibriSpeech corpus](http://www.openslr.org/12/) for speaker identification task. Find  more details in **Dataset preprocessing** section.
  
  * #### Weights of Nsynth model pretrained 
  
  eigther on the [NSynth dataset](https://magenta.tensorflow.org/datasets/nsynth) 
  of individual instrument notes (download weights from [here](http://download.magenta.tensorflow.org/models/nsynth/wavenet-ckpt.tar))
  or dataset with voices (download weights from [here](http://download.magenta.tensorflow.org/models/nsynth/wavenet-voice-ckpt.tar.gz)).
  
 ## Example Usage
 (finetuned encoder)
 
 ```python  nsynth_finetuned_encoder.py```
 
 (finetuned autoencoder)
 
 ```python  nsynth_finetuned_autoencoder.py```
 
 # Exploring embeddings (saving ang generating)
 To run experiments from this section you need Magenta package to be installed
 ([installation guide](https://github.com/tensorflow/magenta)).
 
 To investigate more precisely embeddings learnt with Nsynth model, we simplified a bit a code from Google Magenta team.
 You can save and generate embeddings using pretrained Nsynth model in the following way.
 
 ## Example Usage
(save embeddings)
```
python nsynth_save_embeddings.py
```

(generate embeddings)
```
python nsynth_generate.py
```

# Dataset preprocessing

## Data augmentation
If initial dataset doesn't contain enough samples to train, we augment data by adding addtive/multiplicative/both(additive and multiplicative) noise to audio signals. We assumе that the noise has one of the following distributions: Gaussian, Laplasian or Gaussian mixture. For more details see `augment.py`file.

## Creating TFRecords
For each data we create training and validation datasets in .tfrecord format (see, for example, `librispeech_to_tfrecords.py`, `esc_to_tfrecords.py` files).

