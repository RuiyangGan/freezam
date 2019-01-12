# A function library to conduct time-frequency analysis
# based on short-time Fourier transform

# Import required packages
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt


def stft(signal, bandwidth, window_centers, sampling_rate=1,
         window_type="hann", smoothing=True):
    '''This function calculates short-time fourier transform
    of a given time series object.

    Required functions: window(), periodogram(), smoothing_periodogram()

    Keyword arguments:
    signal -- A signal to be performed Short-Time Fourier Transform

    bandwidth -- A numerical value to specify bandwidth of the
    lag window operator

    window_type -- A string to specify the type of window function used
    (default "hann", the hanning window)

    smoothing -- A boolean value to indicate whether we should smooth the local
    periodograms (default False)

    window_centers -- A list specifying the center of windowed time series
    '''
    windowed_signal = window(signal, window_centers=window_centers,
                             bandwidth=bandwidth, window_type=window_type,
                             sampling_rate=sampling_rate)
    p = int(len(window_centers))
    f = int(sampling_rate*bandwidth/2+1)
    spectrogram = np.empty([f, p])
    frequency = np.empty([f, p])
    for c in range(p):
        freq, power = periodogram(windowed_signal[:, c], sampling_rate)
        spectrogram[:, c] = power
        frequency[:, c] = freq
    if smoothing:
        frequency, spectrogram = smoothing_periodogram(spectrogram,
                                                       time_width=bandwidth)
    return([frequency, spectrogram])


def window(signal, window_centers, bandwidth,
           sampling_rate, window_type="hann"):
    ''' This function turns a time series object into a windowed
    time series object using specified lag window operator with
    given bndwidth

    Keyword arguments:
    signal -- A signal in time domain to be turned into
    a windowed time series

    window_centers -- An array or list indicating centers of the windowed
    time series

    bandwidth -- A numerical value to specify bandwidth of the
    lag window operator

    sampling_rate -- Sampling rate of the signal

    type -- A string to specify the type of lag-window operator used
    (default "hann", the hanning window)
    '''
    p = len(window_centers)
    n = len(signal)
    h = bandwidth
    l = sampling_rate
    T = n/l    # Calculate time length of the audio file in seconds
    windowed_signal = np.zeros([h*l+1, p])
    window_weights = scipy.signal.get_window(window_type, h*l+1)
    for j in range(p):
        t = window_centers[j]
        # Zero padded the sub-signal if the time is at boundary
        # If it is near the start (t-h/2 <= 0), then zero-padded the LHS;
        # If it is near the end (t+h/2 > T), then zero-padded RHS
        if t-h/2 <= 0:
            sub_signal = list(signal[:(int(t+h/2)*l+1)])
            zero_pads = [0 for d in range(h*l+1-len(sub_signal))]
            sub_signal = zero_pads + sub_signal
        elif t+h/2 >= T:
            sub_signal = list(signal[int(np.floor((t-h/2)*l)):])
            zero_pads = [0 for d in range(h*l+1-len(sub_signal))]
            sub_signal = sub_signal + zero_pads
        else:
            sub_signal = signal[int(np.floor((t-h/2)*l)):
                                int(np.ceil((t+h/2)*l)+1)]
        windowed_signal[:, j] = np.multiply(sub_signal, window_weights)
    return(windowed_signal)


def periodogram(signal, sampling_rate):
    '''This function calulates the periodogram of a given
    signal and return the sampling rate and

    Keyword arguments:

    signal -- A time series signal to be turned into
    a windowed time series
    '''
    return(scipy.signal.periodogram(signal, fs=sampling_rate,
           scaling='spectrum'))


# Smooth the periodogram using the flat top kernel
def smoothing_periodogram(spectra, time_width,
                          spectral_width=20, smoothing_type="flattop"):
    '''This function smoothes a given periodogram.

    Keyword arguments:

    periodogram -- A given periodogram to smooth.

    smoothing_type -- A string indicating how to smooth the periodogram
    (default "flattop", use flattop spectral kernal for smmothing)
    '''
    n, p = spectra.shape
    h_freq = spectral_width*time_width     # Set the spectral bandwidth in hz
    spectra_window = smoothing_type
    spec_window_centers = np.arange(start=h_freq, stop=n-h_freq,
                                    step=5*time_width)
    smoothed_periodogram = np.empty([spec_window_centers.shape[0], p])
    binned_freq = np.empty([spec_window_centers.shape[0], p])
    for t in range(p):
        binned_freq[:, t] = spec_window_centers/time_width
        # Obtained the smoothed periodogram
        windowed_prdgm = window(spectra[:, t], spec_window_centers,
                                bandwidth=h_freq, sampling_rate=1,
                                window_type=smoothing_type)
        smoothed_periodogram[:, t] = np.sum(windowed_prdgm, axis=0)
    # Return the smoothed periodogram and frequency
    return [binned_freq, smoothed_periodogram]


def mag_signature(spectrogram, frequency):
    '''This function calculates signature of the Short-Time
    Fourier Transform of a time series. For now, I will use
    method of binning frequencies into nonoverlapping frequency
    bins and record the maximum magnitude within these frequency
    bins

    Keyword arguments:

    spectrogram --  A signal's spectogram(STFT)

    frequency -- The frequency matrix of the spectrogram
    '''
    # Divide the filtered frequency by frequency bins
    # [20,50), [50, 100), [100, 200), [200,400),[400,800),[800,1600),
    # [1600,3200), [3200,6400), [6400, 12800)
    freq_bin = [20, 50, 100, 200, 400, 800, 1600,
                3200, 6400, 12800]
    f, p = spectrogram.shape
    signature = np.zeros([9, p])
    for c in range(p):
        # Assign an index to each element in the periodogram and
        bin_indices = np.digitize(frequency[:, c], freq_bin)
        for j in range(9):
            signature[j, c] = np.max(spectrogram[:, c][bin_indices == (j+1)])
        # Normalize magnitudes in each column so that their range is
        # in [0, 1]
        signature[:, c] /= max(signature[:, c])
    return(signature)


def constellation_map(spectrogram, frequency, prominence=0.3):
    '''This function constructs a constellation map based on the spectogram
    and frequency provided. The constellation map is consisited of local maxima
    in the local periodogram. It will return a list of time-frequency
    coordinates indicating the location of the peaks.

    Keyword Arguement:
    spectrogram --  A signal's spectogram

    frequency -- The frequency matrix of the spectrogram

    prominence -- An optional argument for peak finding, the prominence of
    a peak
    '''
    coordinates = []
    for p in range(spectrogram.shape[1]):
        idx = scipy.signal.find_peaks(spectrogram[:, p] / max(1e-16,
                                      max(spectrogram[:, p])),
                                      prominence=prominence)[0]
        coordinates.append(list(zip([p for i in range(0, len(idx))],
                                    frequency[idx, p])))
    return [x for sublist in coordinates for x in sublist]


def constellation_signature(coordinates, window_centers, time_strip=2):
    '''This function compute the signature from a list of coordinates generated
    from a constellation map.

    Keyword Argument:

    coordinates -- Time-frequency coordinates in a constellation map where
    each coordinate specifies a local maxima in the spectrogram

    time_strip -- The length of the time strip to be included in the
    calculation of the signature
    '''
    signature = np.zeros([len(window_centers), 48])
    # The signature will be composed of number of peaks in the specified
    # frequency bin
    freq_bin = [20, 35, 50, 60, 70, 80, 100, 125,
                150, 175, 200, 225, 250, 275,
                300, 330, 360, 400, 430, 460,
                490, 530, 575, 600, 630, 660,
                700, 750, 800, 850, 900, 950,
                1000, 1100, 1200, 1400, 1600,
                1800, 2000, 2200, 2400, 2600,
                2800, 3000, 3200, 3600, 4000, 4800, 5600]
    (time_loc, freq_loc) = zip(*coordinates)
    time_loc = np.array(time_loc)
    # Assign a bin index to every points in the coordinates
    freq_bin_indices = np.digitize(freq_loc, freq_bin)
    # For every time_strip (following three interval counting itself),
    # calculate the 48-dimensional vector that counts
    # the number of peaks in the specified frequency
    for t in sorted(set(time_loc)):
        signature[t, :] = [sum([int(i) for i in
                               abs(time_loc[freq_bin_indices == j] - t) <
                               time_strip])
                           for j in range(1, 49)]
    return signature


def spectrogram_plot(spectrogram, frequency, time_points):
    '''This function plots the spetrogram of the image in time-frequency
    domain in the form of heat map. The natural log of the spectogram
    will be plotted and return

    Keyword Arguement:

    spectrogram -- A given spectrogram for plot

    frequency -- The frequency of spectrogram

    time_points -- The time points of spectrogram
    '''
    plt.pcolormesh(time_points, frequency, np.log(spectrogram))
