"""Defines small data structures for representing spike trains."""

import numpy as np

class Neuron(object):
    """Represent spike times during events in a single neuron.

    This lets you easily convert a collection of spike times (in seconds)
    into a numpy array or an MNE Epochs object for further processing and
    visualization.

    Parameters
    ----------
    spiketimes : list of arrays, shape (n_events, n_spikes_per_event)
        A list of spiketimes, each item in the list corresponds to one event.
        Each item should be a numpy array (or list) corresponding to the time
        of each spike (in seconds).
    sfreq : float
        The sampling frequency of the spikes.
    tmin : float
        The minimum time when spikes are converted to a timeseries.
    tmax : float
        The maximum time when spikes are converted to a timeseries.
    name : string
        The name of this neuron.
    events : array, shape (n_events,)
        The type of each event. This can be integers or strings.

    Attributes
    ----------
    spikes : array, shape (n_events, n_times), dtype int
        A boolean array corresponding to the spikes for each timepoint.
        If multiple spikes fall within a single time bin (e.g., because
        sfreq is relatively low), then they will be summed together.
    """
    def __init__(self, spiketimes, sfreq=1e3, tmin=None, tmax=None, name=None,
                 events=None):

        if not isinstance(spiketimes[0], (list, np.ndarray)):
            spiketimes = [spiketimes]
        self.spiketimes = spiketimes
        self.max_spiketime = np.max([max(ii) for ii in spiketimes])
        self.min_spiketime = np.min([min(ii) for ii in spiketimes])
        self.n_epochs = len(self.spiketimes)
        self.sfreq = sfreq

        # Handle time
        tmin = np.min([0, self.min_spiketime]) if tmin is None else tmin
        tmax = self.max_spiketime if tmax is None else tmax

        if tmax < tmin:
            raise ValueError('tmax must be greater than or equal to tmin')
        if tmax < self.max_spiketime:
            raise ValueError('tmax must be greater than the max spike time')
        if tmin > self.min_spiketime:
            raise ValueError('tmin must be less than the minimum spike time')
        time = np.arange(int(tmin * sfreq), int(tmax * sfreq))
        self.time = time / float(sfreq)
        self.tmin = tmin
        self.tmax = tmax

        self.name = name
        if events is not None:
            events = np.atleast_1d(events)
            if events.ndim != 1:
                raise ValueError('Events must have a single dimension')
            unique_events = np.unique(events)
            event_id = {ev_id: ii for ii, ev_id in enumerate(unique_events)}
        else:
            events = np.ones(len(spiketimes))
            event_id = {'1': 1}
        self.events = events
        self.event_id = event_id

    def __repr__(self):
        s = 'Name: {} | Num Events: {} | Events: {} | tmin/tmax: ({}, {})'.format(
        self.name, len(self.events), list(self.event_id.keys()), self.tmin, self.tmax)
        return s

    @property
    def spikes(self):
        data = np.zeros([self.n_epochs, len(self.time)])
        for ii, trial in enumerate(self.spiketimes):
            # Correct for the epoch start
            trial_shifted = np.array(trial) - self.tmin
            for spike in trial_shifted:
                data[ii, int(spike * self.sfreq) - 1] += 1
        return data

    def to_mne(self):
        """Convert the spikes into an MNE Epochs object.

        MNE Epochs objects have several useful plotting etc methods. This
        returns such an object so that you can do further processing and viz.
        """
        try:
            import mne
        except ModuleNotFoundError:
            raise ModuleNotFoundError('MNE is not installed.')
        info = mne.create_info([str(self.name)], sfreq=self.sfreq, ch_types='misc')
        data = self.spikes[:, np.newaxis, :]  # Add a singleton channel dimension
        # If events are given in strings, convert to integers first
        events = self.events
        if isinstance(events[0], str):
            events = [self.event_id[event] for event in events]
        events = np.column_stack([range(len(events)), np.zeros_like(events), events])
        events = events.astype(int)
        epochs = mne.EpochsArray(data, info, events=events,
                                 event_id=self.event_id, tmin=self.tmin)
        return epochs
