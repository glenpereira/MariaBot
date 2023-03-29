# MariaBot

A Malayalam TTS engine built using Tacotron2 and WaveGlow.

## Installation

1. Clone the Nvidia Tacotron2 repo:  `git clone --recursive https://github.com/NVIDIA/tacotron2.git`
2. `cd tacotron2/waveglow && git checkout 2fd4e63`
3. Install packages from requirements.txt
4. Install PyTorch: `pip3 install torch torchvision torchaudio`

Credits to [Parapsychic](https://github.com/parapsychic/tacotron2-malayalam) for training the model on the Malayalam language using [Tacotron2](https://github.com/NVIDIA/tacotron2) and [Waveglow](https://github.com/NVIDIA/waveglow/)
