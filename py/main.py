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

sys.path.append(join('tacotron2/', 'waveglow/'))
sys.path.append('waveglow')

force_download_TT2 = True
tacotron2_pretrained_model = './models/Maria'  # @param {type:"string"}
# @param {type:"string"}
waveglow_pretrained_model = './models/waveglow_256channels_ljs_v3.pt'


from hparams import create_hparams


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

def create_audio(input_text):
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

    text = input_text  # @param {type:"string"}
    sigma = 0.8
    denoise_strength = 0.324
    # disables automatic ARPAbet conversion, useful for inputting your own ARPAbet pronounciations or just for testing.
    raw_input = False
    completion_status = False

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


