# A function factory for the input output
import os
import re
from pydub import AudioSegment
import wave
import struct
import urllib.request
import logging
import logging.config

# Create a logger
logging.config.fileConfig("./logging.conf")
logger = logging.getLogger(__name__)


def audio_to_signal(audio_file_source):
    '''This function takes strings that either specify a path
    points to a local audio file or a url string that points to
    a remote audio file. It will identify whether it is a local file
    or a remote url string then choose proper function to read the
    local file or the remote file. After it finds the file, then it
    will convert the audio file into .wav file.

    Required functions: local_to_wav(), url_to_wav()

    Keyword arguments:
    local_audio_path -- A string that specify path of teh local audio file
    '''
    # determine if the audio is remote or local by checking
    # the checking
    urlRE = re.compile("^https?:")
    localRe = re.compile("[.](ogg|mp3|mp4|aiff|flv|wav|wma|flac)$")
    url_flag = urlRE.search(audio_file_source) is not None
    local_flag = localRe.search(audio_file_source) is not None

    if url_flag:
        return(url_to_signal(audio_file_source))
    elif local_flag:
        return(local_to_signal(audio_file_source))
    else:
        logger.error("The file " + audio_file_source +
                     "doesn't have a readable format")
        raise ValueError("The file doesn't have a readable format")


def url_to_signal(remote_url_string):
    '''This function read in a url that points to a remote audio file
    and convert to .wav file

    Keyword arguments:
    remote_url_string -- A string that specifies the url that points
    to a remote audio file
    '''
    file_name, headers = urllib.request.open(remote_url_string)
    # Use local_to_signal() to convert the downloaded file to
    # audio signal
    local_to_signal(file_name)


def local_to_signal(audio_path):
    ''' This function convert local audio file into a wav file,
    and it convert the wav file to a signal over time. If the local
    audio file is in supported format

    Keyword arguments:
    local_audio_path -- A string that specifies the path of the
    audio file
    '''
    # If the file is a .wav file, then we don't need to convert
    # the file into .wav format; if it is not, then we will have
    # to convert it into .wav format
    support_Format = {'.wav': AudioSegment.from_wav,
                      '.mp3': AudioSegment.from_mp3,
                      '.ogg': AudioSegment.from_ogg,
                      '.flv': AudioSegment.from_flv,
                      '.flac': lambda s: AudioSegment.from_file(s, "flac")}
    # Obtain the base path of the file
    audio_base_path = os.path.basename(audio_path)
    # Obtain filename and corresponding extension
    audio_name, audio_ext = os.path.splitext(audio_base_path)
    # Obtain the audio file and convert into .wav format
    audio = support_Format[audio_ext](audio_path)
    audio_wav = audio_name+'.wav'
    # Set stereo to mono
    audio = audio.set_channels(1)
    audio.export('music/'+audio_wav, format='wav')
    # Open the .wav file
    wav = wave.open('music/'+audio_wav, mode='rb')
    # Get the sampling rate of the audio
    sampling_rate = wav.getframerate()
    # Convert the binary chunks to short
    astr = wav.readframes(wav.getnframes())
    # convert binary chunks to short
    audio_signal = struct.unpack("%ih" %
                                 (wav.getnframes() * wav.getnchannels()),
                                 astr)
    audio_signal = [float(val) / pow(2, 15) for val in audio_signal]
    audio_info = {'audio_signal': audio_signal, 'audio_name': audio_name,
                  'audio_source': audio_path, 'sampling_rate': sampling_rate}
    logger.info("Convert the file "+audio_path+" to float amplitude")
    return audio_info
