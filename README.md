# BEAMFORMERS
This library implements some of the most well known beamformers for source separation and speech enhancement.
The beamformers are easy to use and implemented in simplest way possible in case someone wants to understand 
how to implemented them on their own. The idea of this library is to provide a simple way of applying beamforming for 
source separation and/or speech enhancement directly to multi-channel microphone recordings store as numpy arrays.

For most of the beamformers the only information needed is the microphone recordings of the mixture 
(or the noisy speech) and a segment of recordings of the noise alone. No need to provide the steering vector as 
it is automatically extracted from the data (see docs for more information on how this works). If available, 
recordings of the target speech alone will help the estimate of the steering vector providing a cleaner output.

## Install
Simply
```
pip install beamformers
```
or 
```
git clone https://github.com/Enny1991/beamformers
cd beamformers
python setup.py
```

## Simple to use 
```
import soundfile as sf
from beamformers import beamformers

# mix: ndarray (n_mics, time) with the noisy signal recording
# nn: ndarray (n_mics, time) with the noise recording

spk, _ = sf.read('../wavs/spk.wav')
# soundfile loads the file is as (time, n_mics) so I need to transpose it
spk = spk.T

# NOISE
nn, _ = sf.read('../wavs/nn.wav')
# soundfile loads the file is as (time, n_mics) so I need to transpose it
nn = nn.T

out_mvdr = beamformers.MVDR(mix, nn)
```