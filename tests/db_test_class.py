import pytest
from DbManager import (signature_match, library_add,
                       library_read, library_delete,
                       library_update_song, library_update_title)
# Import the Song class
from .Song import Song
import psycopg2


# Add a few stub songs for testing purpose
# The first and second song has the same signal
# but with different names
# The third song are substantively different from
# the first two as they have different signals and names
t = range(1000)
centers = [x for x in range(1000) if x % 4 == 0]
signal_1 = np.sin(2*np.pi*t)
song_1 = Song(signal=signal_1, title_info="Sine Wave",
              sampling_rate_info=500,
              window_centers_info=centers,
              start_time_info=0, length_info=1000)

song_2 = Song(signal=signal_1, title_info="Sine Wave 2",
              sampling_rate_info=500,
              window_centers_info=centers,
              start_time_info=0, length_info=1000)
signal_2 = np.random.normal(size=1000)
song_3 = Song(signal=signal_2, title_info="White Noise",
              sampling_rate_info=500,
              window_centers_info=centers,
              start_time_info=0, length_info=1000)

# Add the first song in the library now
library_add(song_1)


def library_add_existed():
    # add an existed song into the library. DbManager
    # should catch the error occured in postgres for
    # adding the song with same title
    with pytest.raises(psycopg2.DatabaseError):
        library_add(song_1)


def test_library_read():
    # Since the song table should have only 1 stub song,
    # the number of song in the database should be 1
    rows = library_read()
    assert len(rows) == 1


def library_match_existed():
    # Match a signature for an entire song in the Database
    # It should return the match
    matched_song, matched_time, params_cp, T, acy_level = signature_match(
        song_1.signature)
    assert matched_song == 'Sine Wave'


def update_title_not_existed_song():
    # Update a song that doesn't exist in the library.
    # The database manager should catch the exception
    # where there is no corresponding title
    with pytest.raises(psycopg2.DatabaseError):
        library_update_title('Sine Wave 2', 'Make_up_Name')


def update_song_not_existed_song():
    # Update a song that doesn't exist in the library.
    with pytest.raises(psycopg2.DatabaseError):
        library_update_song('Sine Wave 2', song_2)


def delete_not_existed_song():
    # When the title provided is none existent in the database
    # the library should not delete any song. Therefore, we will
    # test on whether the program will delete songs that should
    # not be deleted
    rows_before_deletion = len(library_read())
    # delete with non-existent title
    library_delete('None_Existent')
    rows_after_deletion = len(library_read())
    assert rows_after_deletion == rows_before_deletion


def library_match_different():
    # When there are two drastically different song in the
    # database, the program should recognize the right song
    # Add the song that represents white noise
    library_add(song_3)
    # If added successfully, the following row retrieved should
    # be 1, we will use for assetion later
    row = library_read('White Noise')
    # Now search for a match for song_1, which is the sine wave
    matched_song, matched_time, params_cp, T, acy_level = signature_match(
        song_1.signature)
    # Now check for both the match of signature and the addition of
    # white noise in the library
    assert row == 1 and matched_song == 'Sine Wave'
