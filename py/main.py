import sys
from tacotron2.waveglow.denoiser import Denoiser
from tacotron2.text import text_to_sequence
from tacotron2.audio_processing import griffin_lim
from tacotron2.layers import TacotronSTFT
from tacotron2.model import Tacotron2
import torch
import numpy as np
import matplotlib.pylab as plt
import matplotlib
import time
import os
from os.path import exists, join, basename, splitext
from scipy.io.wavfile import write

import tensorflow as tf


class HParams(object):
    hparamdict = []

    def __init__(self, **hparams):
        self.hparamdict = hparams
        for k, v in hparams.items():
            setattr(self, k, v)

    def __repr__(self):
        return "HParams(" + repr([(k, v) for k, v in self.hparamdict.items()]) + ")"

    def __str__(self):
        return ','.join([(k + '=' + str(v)) for k, v in self.hparamdict.items()])

    def parse(self, params):
        for s in params.split(","):
            k, v = s.split("=", 1)
            k = k.strip()
            t = type(self.hparamdict[k])
            if t == bool:
                v = v.strip().lower()
                if v in ['true', '1']:
                    v = True
                elif v in ['false', '0']:
                    v = False
                else:
                    raise ValueError(v)
            else:
                v = t(v)
            self.hparamdict[k] = v
            setattr(self, k, v)
        return self


def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = HParams(
        ################################
        # Experiment Parameters        #
        ################################
        epochs=500,
        iters_per_checkpoint=1000,
        seed=1234,
        dynamic_loss_scaling=True,
        fp16_run=False,
        distributed_run=False,
        dist_backend="nccl",
        dist_url="tcp://localhost:54321",
        cudnn_enabled=True,
        cudnn_benchmark=False,
        ignore_layers=['embedding.weight'],

        ################################
        # Data Parameters             #
        ################################
        load_mel_from_disk=False,
        training_files='filelists/ljs_audio_text_train_filelist.txt',
        validation_files='filelists/ljs_audio_text_val_filelist.txt',
        text_cleaners=['transliteration_cleaners'],

        ################################
        # Audio Parameters             #
        ################################
        max_wav_value=32768.0,
        sampling_rate=22050,
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=8000.0,

        ################################
        # Model Parameters             #
        ################################
        n_symbols=148,
        symbols_embedding_dim=512,

        # Encoder parameters
        encoder_kernel_size=5,
        encoder_n_convolutions=3,
        encoder_embedding_dim=512,

        # Decoder parameters
        n_frames_per_step=1,  # currently only 1 is supported
        decoder_rnn_dim=1024,
        prenet_dim=256,
        max_decoder_steps=1000,
        gate_threshold=0.5,
        p_attention_dropout=0.1,
        p_decoder_dropout=0.1,

        # Attention parameters
        attention_rnn_dim=1024,
        attention_dim=128,

        # Location Layer parameters
        attention_location_n_filters=32,
        attention_location_kernel_size=31,

        # Mel-post processing network parameters
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,

        ################################
        # Optimization Hyperparameters #
        ################################
        use_saved_learning_rate=False,
        learning_rate=1e-3,
        weight_decay=1e-6,
        grad_clip_thresh=1.0,
        batch_size=32,
        mask_padding=True  # set model's padded outputs to padded values
    )

    if hparams_string:
        tf.compat.v1.logging.info(
            'Parsing command line hparams: %s', hparams_string)
        hparams.parse(hparams_string)

    if verbose:
        tf.compat.v1.logging.info('Final parsed hparams: %s', hparams.values())

    return hparams


sys.path.append(join('tacotron2/', 'waveglow/'))
sys.path.append('waveglow')

force_download_TT2 = True
tacotron2_pretrained_model = './models/Maria'  # @param {type:"string"}
# @param {type:"string"}
waveglow_pretrained_model = './models/waveglow_256channels_ljs_v3.pt'


# from hparams import create_hparams


#!gdown --id '1E12g_sREdcH5vuZb44EZYX8JjGWQ9rRp'
thisdict = {}
# for line in reversed((open('merged.dict.txt', "r").read()).splitlines()):
#    thisdict[(line.split(" ",1))[0]] = (line.split(" ",1))[1].strip()


def ARPA(text):
    out = ''
    for word_ in text.split(" "):
        word = word_
        end_chars = ''
        while any(elem in word for elem in r"!?,.;") and len(word) > 1:
            if word[-1] == '!':
                end_chars = '!' + end_chars
                word = word[:-1]
            if word[-1] == '?':
                end_chars = '?' + end_chars
                word = word[:-1]
            if word[-1] == ',':
                end_chars = ',' + end_chars
                word = word[:-1]
            if word[-1] == '.':
                end_chars = '.' + end_chars
                word = word[:-1]
            if word[-1] == ';':
                end_chars = ';' + end_chars
                word = word[:-1]
            else:
                break
        try:
            word_arpa = thisdict[word.upper()]
        except:
            word_arpa = ''
        if len(word_arpa) != 0:
            word = "{" + str(word_arpa) + "}"
        out = (out + " " + word + end_chars).strip()
    if out[-1] != ";":
        out = out + ";"
    return out

# torch.set_grad_enabled(False)


# initialize Tacotron2 with the pretrained model
hparams = create_hparams()

# @title Parameters
# Load Tacotron2 (run this cell every time you change the model)
hparams.sampling_rate = 22050  # Don't change this
# How long the audio will be before it cuts off (1000 is about 11 seconds)
hparams.max_decoder_steps = 1000
# Model must be 90% sure the clip is over before ending generation (the higher this number is, the more likely that the AI will keep generating until it reaches the Max Decoder Steps)
hparams.gate_threshold = 0.1
model = Tacotron2(hparams)
model.load_state_dict(torch.load(tacotron2_pretrained_model)['state_dict'])
_ = model.cuda().eval().half()

# Load WaveGlow
waveglow = torch.load(waveglow_pretrained_model)['model']
waveglow.cuda().eval().half()
for k in waveglow.convinv:
    k.float()
denoiser = Denoiser(waveglow)

text = 'ningaluTe aappil ee nampar aTikkuka.'  # @param {type:"string"}
sigma = 0.8
denoise_strength = 0.324
# disables automatic ARPAbet conversion, useful for inputting your own ARPAbet pronounciations or just for testing.
raw_input = False
completion_status = False


def create_audio(text):
    for i in text.split("\n"):
        if len(i) < 1:
            continue
            print(i)
        if raw_input:
            if i[-1] != ";":
                i = i+";"
        else:
            i = ARPA(i)
        print(i)
        with torch.no_grad():  # save VRAM by not including gradients
            sequence = np.array(text_to_sequence(i, ['basic_cleaners']))[None, :]
            sequence = torch.autograd.Variable(
                torch.from_numpy(sequence)).cuda().long()
            mel_outputs, mel_outputs_postnet, _, alignments = model.inference(
                sequence)
    # plot_data((mel_outputs_postnet.float().data.cpu().numpy()[0],
    #          alignments.float().data.cpu().numpy()[0].T))
        # print(""); #ipd.display(ipd.Audio(audio[0].data.cpu().numpy(), rate=hparams.sampling_rate))
            audio = waveglow.infer(mel_outputs_postnet, sigma=sigma)
            audio_numpy = audio[0].data.cpu().numpy().astype(np.float32)
            write("sample.wav", hparams.sampling_rate, audio_numpy)
            completion_status = True
            print(audio_numpy)
    
    return completion_status

create_audio(text)

