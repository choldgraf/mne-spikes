"""Defines small data structures for representing spike trains."""

import numpy as np
import mne

class Neuron(object):
    def __init__(self, spiketimes, sfreq=1e3, length=None, name=None,
                 events=None, event_id=None):
        if not isinstance(spiketimes[0], (list, np.ndarray)):
            spiketimes = [spiketimes]
        self.spiketimes = spiketimes
        self.max_spiketime = np.max([max(ii) for ii in spiketimes])
        self.n_epochs = len(self.spiketimes)
        self.sfreq = sfreq
        self.length = self.max_spiketime if length is None else length
        if self.length < self.max_spiketime:
            raise ValueError('length must be greater than or equal to the max spiketime')
        self.time = np.arange(int(sfreq * self.length)) / float(sfreq)
        self.name = name
        # XXX events shouldn't have to be the same length as spiketimes
        if events is not None:
            events = np.atleast_1d(events)
            if events.ndim != 1:
                raise ValueError('Events must have a single dimension')
            if len(events) != len(spiketimes):
                raise ValueError('Events must be the same length as spiketimes')

        self.events = events
        self.event_id = event_id

    @property
    def spikes(self):
        data = np.zeros([self.n_epochs, self.time.shape[0]])
        for ii, trial in enumerate(self.spiketimes):
            for spike in trial:
                data[ii, int(spike * self.sfreq) - 1] += 1
        return data

    def to_mne(self, psth=True):
        info = mne.create_info([self.name], sfreq=self.sfreq, ch_types='misc')
        data = self.spikes[:, np.newaxis, :]  # Add a singleton channel dimension
        if self.events is not None:
            if isinstance(self.events[0], str):
                unique_events = set(self.events)
                event_id = {event: ii for ii, event in enumerate(unique_events)}
                events = [event_id[event] for event in self.events]
            else:
                events = self.events
            events = np.column_stack([range(len(events)), np.zeros_like(events), events])
        else:
            events = None
        if psth is True:
            data = data / len(data)
        epochs = mne.EpochsArray(data, info, events=events, event_id=event_id)
        return epochs
