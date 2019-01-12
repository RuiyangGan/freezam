# Import Short-Time-Fourier-Transform Module
from STFT import *
import numpy as np


class Song:
    """ A Song Class that captures important attributes of a song

    Instance variables:
    title -- A Song's title
    sampling_rate -- The song's sampling rate
    artist -- The artist affiliated with the song
    spectrogram -- The song's local periodogram
    signature -- A song's specific signature calculated by
    """

    def __init__(self, signal, title_info, sampling_rate_info,
                 artist_info=None, date_info=None, album_info="",
                 address_info="", start_time_info=0, window_type_info="hann"):
        self.title = title_info
        self.date = date_info
        self.album = album_info
        self.address = address_info
        if artist_info is None:
            self.artist = 'VA'

        # Set songs' sampling rate and calculate its spectrogram
        # according to the signal, window center, window type,
        # sampling rate, bandwidth. Using the song's spectrogram,
        # Calculate the signature
        self.sampling_rate = sampling_rate_info
        self.start_time = start_time_info
        self.T = len(signal)/self.sampling_rate
        self.window_type = window_type_info
        self.window_centers = np.arange(start=0, stop=self.T+1,
                                        step=1)
        self.freq, self.power = stft(signal, bandwidth=10,
                                     sampling_rate=self.sampling_rate,
                                     window_centers=self.window_centers,
                                     window_type=self.window_type)
        self.constell_map = constellation_map(self.power, self.freq)
        self.signature = constellation_signature(self.constell_map,
                                                 self.window_centers)

    # Instance methods for getting and setting attributes of a song object
    def set_title(self, title_info):
        self.title = title_info

    def set_artist(self, artist_info):
        self.artist_info = artist_info

    def set_signature(self, signature_info):
        self.signature = signature_info

    def set_spectrogram(self, spectrogram):
        self.spectrogram = spectrogram

    def set_frequency(self, frequency):
        self.frequency = frequency

    def set_key(self, key_info):
        self.key = key_info

    def get_title(self):
        return(self.title)

    def get_artist(self):
        return(self.artist)

    def get_signature(self):
        return(self.signature)

    def get_key(self):
        return(self.key)

    def get_address(self):
        return(self.address)

    def get_length(self):
        return(self.T)

    def get_spectrogram(self):
        return(self.power)

    def get_frequency(self):
        return(self.freq)

    def spectogram_image(self):
        spectrogram_plot(self.power, self.freq[:, 0], self.window_centers)
