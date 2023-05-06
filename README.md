# MariaBot

A Malayalam TTS engine built using Tacotron2 and WaveGlow.

## Installation

1. Clone the malayalam-tacotron2 repo:  `git clone --recursive https://github.com/parapsychic/malayalam-tacotron2`
2. `cd malayalam-tacotron2/waveglow && git checkout 2fd4e63`
3. Rename the cloned folder to "malayalam_tacotron2"
4. Change the path appendation in malayalam_tacotron2/waveglow/denoiser.py line 2
5. Initialize a new venv
6. Install packages from requirements.txt
7. Install PyTorch: `pip3 install torch torchvision torchaudio`
8. Configure aws cli with access keys

Credits to [Parapsychic](https://github.com/parapsychic/tacotron2-malayalam) for training the model on the Malayalam language using [Tacotron2](https://github.com/NVIDIA/tacotron2) and [Waveglow](https://github.com/NVIDIA/waveglow/)
