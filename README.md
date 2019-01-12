36-650 Challange project for Shazam. 
Author: Ruiyang Gan

Dependencies:
FALCONN-LIB
pydub
ffmpeg
scipy
postgres 10.0
(Ideally in a linux environment)


This is the Challange project for 36-650 stats computing. The Challange project
is Shazam. It uses STFT to represent the time-frequency diagragm of an audio
and record robust constellations (the peaks in the energy levels of the diagram
at each time point). Using this robust constellations, we record number of peaks in 
40 non-overlapping frequency bins at each time (and possibly a few seconds following). 
We use that as a signature of the song. We use pydub and ffmpeg for reading audio files
other than .wav formats.

After the signature of the song is calculated, we store the song and its signature
into seperate tables of postgres database. To identify a snippet and find a match
in the song, we first calculate the signature of the audio and use the locality
sensitive hashing to search snippet's signature match with the song in the database.
(As each signature is a 40-dimensional vector, we can think of it as a 
nearest-neighbor searching problem). 
We use FALCONN-LIB for the implementation of locality-sensitive-hasing and look-up.


To search a snippet that is approximately 15 seconds long in a library with 50 song,
the searching process usually takes around 7~8 seconds. The searching is generally
robust to some distortion of the audio. However, the program tends to under-perform
when the background noise is very loud. Also, for genre with sophiscated composition
and various instruments, the program tends to not search the song intend to but to
search a song that is similar in style. 
