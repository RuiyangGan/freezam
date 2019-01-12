# Import testing framwork and scipy.signal
# package to generate signals for testing
# purpose
import numpy as np
from STFT import (stft, mag_signature, window,
                  periodogram, smoothing_periodogram,
                  constellation_map, constellation_signature)
import scipy.sparse


# Testing the window function
def window_test_bandwidth():

    # Test if bandwidth is set to 0, then time series value
    # at each window center will be returned
    t = range(1000)
    centers = [x for x in range(1000) if x % 4 == 0]
    signal = np.sin(2*np.pi*t)
    assert window(signal, centers, bandwidth=0) == signal[centers]


def window_test_centers():
    # Test if the output, the windowed time series, has
    # number of columns that is same as the size
    # of the window centers passed in as argument
    t = range(1000)
    centers = [x for x in range(1000) if x % 4 == 0]
    signal = np.sin(2*np.pi*t)
    assert np.ncol(window(centers)) == len(centers)


def periodogram_test_neg():
    # Test if the periodogram of a given signal
    # is negative.
    t = range(1000)
    signal = np.sin(2*np.pi*t)
    assert np.all(periodogram(signal) >= 0)


def periodogram_test_WN():
    # The periodogram of the white noise should be constant
    # throughout the frequency domain.
    n = 1000
    mean = 0
    sd = 1
    WN = np.random.normal(mean, sd, n)
    assert np.allclose(periodogram(WN), [1 for d in range(1000)])


def periodogram_test_RW():
    # The periodogram of a random walk should be very close
    # to a spike of infinite degree at zero frequency
    Z = np.random.normal(size=1001)
    X = [0 for d in range(1000)]
    X[0] = 0
    for t in range(1, 1001):
        X[t] = X[t-1] + Z[t]
    assert np.isclose(periodogram(X)[0], np.PINF)


def periodogram_test_sine():
    # The periodogram of a sine wave should have
    # a peak around frequency specified period, and then
    # remains low for other frequencies.
    t = range(999)
    sine_period = 4
    sine_signal = np.sin(sine_period*2*np.pi*t)
    # The periodogram of the selecte sine wave
    # should peak at around 250th location
    peak_interval = range(245, 255)
    # Record the location of maximum of the calculated
    # periodogram of the sine wave
    actual_peak_loc = np.amax(periodogram(sine_signal))
    assert (actual_peak_loc in peak_interval)


def signature_test_RW():
    # The signature of a random walk at the first
    # row for each column (windowed time series at window center)is 0,
    # as the peak always appear at the first location
    Z = np.random.normal(size=1001)
    X = [0 for d in range(1000)]
    X[0] = 0
    for t in range(1, 1001):
        X[t] = X[t-1] + z[t]
    assert np.allclose(mag_signature(RW_obj)[0, :] == 0)


def smoothing_periodogram_test_size():
    # We will first test the simplest case: A simple moving
    # average spectral window. Then the smoothing periodogram
    # is just a local average. In scipy.signal, the equivalent
    # is the boxcar window
    spectra = np.random.normal(size=1e4).reshape(1000, 10)
    time_width = 1
    smoothing_type = "boxcar"
    spectra_width = 20
    rslt = smoothing_periodogram(spectra, time_width
                                 spectral_width, smoothing_type)[1]
    # calculate the brute-force version of local averaging
    rslt_bf = np.empty([193, 10])
    for j in range(10):
        for i in range(193):
            rslt_bf[i, j] = np.mean(spectra[(i-spectra_width):
                                            (i+spectra_width+1), j])
    assert np.allclose(rslt.reshape(-1, 1), rslt_bf.reshape)


def constellation_map_test():
    # We test the constellation map of a white noise
    # Ideally, we should not expect any peaks as they
    # are randomly scattered in time-frequency diagram
    # with similar energy level. Therefore, under such
    # case, I would not expect too many peaks
    WN = np.random.normal(size=1e4).reshape(1000, 10)
    # Calculate the STFT of the random walk signal
    WN_freq, WN_power = stft(WN, bandwidth=10,
                             window_centers=np.arange(10))
    test_constell = constellation_map(WN_power, WN_freq)
    time, freq = zip(*test_constell)
    # Test if there are many peaks
    assert len(time) <= 1e4 * 0.01


def constellation_signature_test():
    # We test the constellation signature of a white
    # noise. From the previous test, we notice that
    # there should not be many peaks in the constellation
    # map. Therefore, the signature matrix should also
    # be a sparse matrix. For this, we will use
    # scipy.sparse.issparse to test whether the matrix
    # is very sparse.
    WN = np.random.normal(size=1e4).reshape(1000, 10)
    # Calculate the STFT of the random walk signal
    WN_freq, WN_power = stft(WN, bandwidth=10,
                             window_centers=np.arange(10))
    test_constell = signature(WN_power, WN_freq)
    test_signature = constellation_signature(test_constell)
    assert scipy.sparse.issparse(test_signature)
