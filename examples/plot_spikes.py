"""
Plot a spike raster with MNE
----------------------------

Converting spike times to an MNE object, then plotting.

First we'll generate a bunch of random spike times, corresponding to
several trials of activity. We'll convert these into an mne-spikes ``Neuron``
object, then convert this to MNE so we can plot.
"""
from mnespikes import Neuron
import numpy as np
import matplotlib.pyplot as plt

sfreq = 100
n_trials = 10
time_mean = .5
spikes_per_trial = [np.random.randint(25, 50) for _ in range(n_trials)]
spiketimes = np.array([.1 * np.random.randn(ii) + time_mean
                       for ii in spikes_per_trial])
# Each item is a list of spike times
print(spiketimes[:2])

################################################################################
# Now we'll convert to a Neuron object. We can easily export this to a NumPy
# array, or to an MNE object.
event_name = 'event!'
neuron = Neuron(spiketimes, sfreq=sfreq, tmin=-.1, tmax=1.5,
                events=[event_name] * len(spiketimes))
print(neuron)
print(neuron.spikes.shape)

################################################################################
# Now convert to an MNE object, which lets us do fancier plotting / analysis.
epochs = neuron.to_mne()
fig = epochs.plot_image([0], show=False)
plt.setp(fig[0].axes[1], ylim=[0, None], ylabel='spikes / s')
plt.show()
