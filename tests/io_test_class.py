# Unit tests for IO module
import pytest
from IO import audio_to_signal, url_to_signal, local_to_signal


# A test for read local audio file with an mp3 extension
def local_file_test():
    # The function shoudld identify this is a local audio file
    # and calls local_to_signal("test.mp3")
    signal = audio_to_signal("test.mp3")
    signal_expected = local_to_signal("mp3")
    assert signal == signal_expected


def remote_file_test():
    # The function should identify this is a url that points
    # to a remote file and calls the function
    # url_to_signal("https://www.test.mp3")
    signal = audio_to_signal("https://www.test.mp3")
    signal_expected = url_to_signal("https://www.test.mp3")
    assert signal == signal_expected


def incorrect_extension_test():
    # audio_to_signal("test.png")should raise an exception and
    # tell the user that the file they provide is not a proper
    # audio file
    with pytest.raises(Exception):
        audio_to_signal("test.png")
