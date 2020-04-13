#!/usr/bin/python3
import numpy as np
from scipy import signal
import sys, wave

# Read a tone from a wave file.
def read_wave(filename):
    w = wave.open(filename, "rb")
    info = w.getparams()
    fbytes = w.readframes(info.nframes)
    w.close()
    sampletypes = {
        1: (np.uint8, -(1 << 7), 1 << 8),
        2: (np.int16, 0.5, 1 << 16),
        4: (np.int32, 0.5, 1 << 32),
    }
    if info.sampwidth not in sampletypes:
        raise IOException()
    sampletype, sampleoff, samplewidth = sampletypes[info.sampwidth]
    samples = np.frombuffer(fbytes, dtype=sampletype)
    scale = 2.0 / samplewidth
    fsamples = scale * (samples + sampleoff)
    channels = np.reshape(fsamples, (-1, info.nchannels))
    return (info, np.transpose(channels))

# Write a tone to a wave file.
def write_wave(filename, info, channels):
    samples = np.reshape(np.transpose(channels), (-1,))
    sampletypes = {
        1: ('u1', (1 << 7), 1 << 8),
        2: ('<i2', 0.5, 1 << 16),
        4: ('<i4', 0.5, 1 << 32),
    }
    if info.sampwidth not in sampletypes:
        raise IOException()
    sampletype, sampleoff, samplewidth = sampletypes[info.sampwidth]
    scale = samplewidth / 2.0
    fsamples = scale * samples + sampleoff
    hsamples = np.array(fsamples, dtype=sampletype)
    w = wave.open(filename, "wb")
    w.setparams(info)
    w.writeframes(hsamples)
    w.close()

wah_rate = 1.0 / float(sys.argv[1])
in_filename = sys.argv[2]
out_filename = sys.argv[3]

info, channels = read_wave(in_filename)

blocksize = info.framerate // 100
window = signal.windows.hann(blocksize)
ncountour = int(wah_rate * info.framerate / blocksize)
countour = 0.3 * np.log2(2 - np.sin(np.linspace(0, np.pi, ncountour))) + 0.01
print(countour)
icountour = 0
waheds = []
for channel in channels:
    wahed = np.zeros(info.nframes)
    start = 0
    while start < info.nframes:
        end = min(start + blocksize, info.nframes)
        block = channel[start:end]
        fw = signal.firwin(128, countour[icountour], window=('kaiser', 0.5))
        icountour = (icountour + 1) % ncountour
        block = signal.convolve(block, fw, mode='same')
        for i in range(start, end):
            wahed[i] += 0.25 * block[i - start] * window[i - start]
        start += blocksize // 4
    waheds.append(wahed)

write_wave(out_filename, info, np.array(waheds))
