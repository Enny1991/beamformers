# BEAMFORMERS [![Build Status](https://travis-ci.org/Enny1991/beamformers.svg?branch=master)](https://travis-ci.org/Enny1991/beamformers.svg?branch=master)

This library implements some of the most well known beamformers for source separation and speech enhancement.
The beamformers are easy to use and implemented in simplest way possible in case someone wants to understand 
how to implemented them on their own. The idea of this library is to provide a simple way of applying beamforming for 
source separation and/or speech enhancement directly to multi-channel microphone recordings store as numpy arrays.

For most of the beamformers the only information needed is the microphone recordings of the mixture 
(or the noisy speech) and a segment of recordings of the noise alone. No need to provide the steering vector as 
it is automatically extracted from the data (see docs for more information on how this works). If available, 
recordings of the target speech alone will help the estimate of the steering vector providing a cleaner output.

Now with also mask-based beamformers!
## Install
Simply
```bash
pip install beamformers
```
or 
```bash
git clone https://github.com/Enny1991/beamformers
cd beamformers
python setup.py install
```

## Simple to use 
```python
import soundfile as sf
from beamformers import beamformers

# mix: ndarray (n_mics, time) with the noisy signal recording
# nn: ndarray (n_mics, time) with the noise recording

mix, _ = sf.read('../wavs/mix.wav')
# soundfile loads the file is as (time, n_mics) so I need to transpose it
mix = mix.T

# NOISE
nn, _ = sf.read('../wavs/nn.wav')
# soundfile loads the file is as (time, n_mics) so I need to transpose it
nn = nn.T

out_mvdr = beamformers.MVDR(mix, nn)
```

## Extra
There is the possibility to use BeamformIt but you will need to manually compile it.
Follow the instructions in their [repo](https://github.com/xanguera/BeamformIt).
Once installed you can use it in the following way 

```python
import soundfile as sf

from beamformers import beamformers

mix, _ = sf.read('../wavs/mix.wav')
mix = mix.T

basedir = '/some/path/BeamformIt'
out_mvdr = beamformers.BeamformIt(mix, fs=8000, basedir=basedir)
```