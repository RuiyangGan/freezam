# A database (music library) manager function library
# I will use pickle to store Song object
import psycopg2
from psycopg2.extras import execute_values
from configparser import ConfigParser
import falconn_util
import h5py
from STFT import *
from collections import Counter
import Song
import numpy as np
import logging
import logging.config
from scipy import spatial


# Create a logger
logging.config.fileConfig("./logging.conf")
logger = logging.getLogger(__name__)


def slowSearch(snippet):
    ''' This function takes a snippet
    searches throuh the library by comparing the local
    periodograms without a special signature.

    Keyword argument:
    snippet -- A given snippet, coming from the IO module
    with the given information about signal, sampling_rate,

    '''
    # Obtain the size of the snippet
    snippet_signal = snippet['audio_signal']
    snippet_name = snippet['audio_name']
    sampling_rate = snippet['sampling_rate']
    snippet_address = snippet['audio_source']
    # Construct a Song Object called snippet_song
    snippet_song = Song.Song(snippet_signal, title_info=snippet_name,
                             address_info=snippet_address,
                             sampling_rate_info=sampling_rate)
    snippet_spectra = snippet_song.get_spectrogram()
    # Deserialize song_lib and search through periodograms in
    # song_lib
    with open('song_lib.pickle', 'rb') as handle:
        song_lib = pickle.load(handle)

    # Search through the whole library and compare the snippet
    # with every song.
    for song in song_lib:
        # Get the song's spectrogram
        hf = h5py.File(song_lib[song]['h5_address'], 'r')
        song_spectra = hf['spectrogram'][()]
        hf.close()
        # Compare the song's spectrogram with the snippet's
        # spectrogram
        for i in range(snippet_spectra.shape[1]-6):
            for j in range(song_spectra.shape[1]-6):
                # Calculate the cosine similarity between local periodogram
                initial_similarity = 1 - spatial.distance.cosine(
                                         snippet_spectra[:, i],
                                         song_spectra[:, j])
                # If the similarity is bigger than 75%, then we will search
                # through the successive window and determine if we had a match
                if initial_similarity >= .75:
                    successive_similarity = [spatial.distance.cosine(
                                             snippet_spectra[:, i+k],
                                             song_spectra[:, j+k])
                                             for k in range(1, 6)]
                    successive_similarity = [1 - s for s in
                                             successive_similarity]
                    if min(successive_similarity) >= 0.7:
                        # matching_rslt = 'The snippet finds a match!'
                        print("""The snippet finds a match!.
                                 The matching song is: \n""")
                        print(song)
                        return(library_read(song))
    print("The snippet doesn't find a match in the library.")


def connect_to_Lib(filename='database.ini', section='freezam'):
    '''This function connects to the song library database.
    I will give most of the credit of writing this function to the
    psycopg2 tutorial on postgresqltutorial.com.Link to tutorial:
    http://www.postgresqltutorial.com/postgresql-python/connect/
    '''
    # Create a parser
    parser = ConfigParser()
    parser.read(filename)

    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'
                        .format(section, filename))

    # Connect to the database
    conn = None
    try:
        conn = psycopg2.connect(**db)
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)

    # return the connection object for operation in the database
    if conn is not None:
        return conn


def library_add(song):
    '''This function add the song and its corresponding signature
    to the database. Return True if the song is successfully
    to the database. It will look for a table called song. It will
    also look for a signature table, where it contains the signature
    at each window center for each song listed in the song_table.

    The structure of the song table will be in the form:
    --------------------------------------------------------
    id|title|artist|sampling_rate|length
    --------------------------------------------------------

    The strucuture of the signature will be in the form:
    --------------------------------------------------------
    title|window_id|window_time|signature
    --------------------------------------------------------
    , where song_id is a foreign key that references ID in the
    song table

    Keyword argument:

    song -- A song object with meta data including: title, artists
    spectrogram, signature, etc.
    '''
    try:
        # Establish a connection to the database
        conn = connect_to_Lib()

        # create cursor for operation in the database
        cur = conn.cursor()

        # Check if a table named song_table exists; if not, create
        # the table with the specified structure and the name song
        cur.execute("""CREATE TABLE IF NOT EXISTS song(
                    id SERIAL,
                    title text PRIMARY KEY,
                    artist text,
                    sampling_rate numeric,
                    length numeric
                        );
                    """)

        # Check if a table called signature exists; if not, create
        # the table with the specified structure and the name signature
        cur.execute("""CREATE TABLE IF NOT EXISTS signature(
                        title text REFERENCES song(title),
                        window_id integer,
                        window_time numeric,
                        signature numeric[]
                        );
                    """)

        # Insert song information into song table
        song_insertion_sql = """INSERT INTO song(title, artist, sampling_rate, length)
                              VALUES(%s, %s, %s, %s)"""
        cur.execute(song_insertion_sql, [song.title, song.artist,
                                         song.sampling_rate, song.T])
        # Insert signature into signature table
        signature_insertion_sql = """INSERT INTO signature(title, window_id,
                                     window_time, signature) VALUES %s"""
        # Convert numpy array to tuples for bulk insertion
        sig_tup = tuple(map(tuple, song.signature))
        num_of_windows = len(song.window_centers)
        sig_ins_dataset = [(song.title, i + 1,
                            song.window_centers[i],
                            "{"+",".join([str(s) for s in sig_tup[i]])+"}")
                           for i in range(num_of_windows)]
        # Bulk insert into signature table
        execute_values(cur, signature_insertion_sql,
                       sig_ins_dataset, template=None)
        # Commit the change and close the cursor
        conn.commit()
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
            raise error
    finally:
        # Close the connection at the end
        if conn is not None:
            conn.close()


def library_delete(song_name):
    '''This funciton delete songs in the library
    The function delete a song in song_lib by the name

    Keyword argument:

    song_name -- A string that indicates the song's name, which specifies
    the key in the song table
    '''
    conn = None
    try:
        conn = connect_to_Lib()
        cur = conn.cursor()

        # First delete rows in signature table with given song title
        cur.execute("DELETE FROM signature WHERE title = %s", [song_name])

        # Then, delete the row in song table
        cur.execute("DELETE FROM song where title = %s", [song_name])

        # commit the change and close cursor
        conn.commit()
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is None:
            conn.close()


def library_read(song_name=None):
    '''This function get the corresponding information of a song
    by its name

    Keyword argument:

    song_name -- The name of the given song
    '''
    conn = None
    try:
        conn = connect_to_Lib()
        cur = conn.cursor()
        if song_name is None:
            cur.execute("""SELECT * FROM song""")
        else:
            cur.execute("SELECT * FROM song WHERE title = %s", [song_name])
        rows = cur.fetchall()
        return rows
    except (Exception, psycopg2.DatabaseError) as error:
        raise error
    finally:
        if conn is not None:
            conn.close()


def library_update_artist(song_name, artist_name):
    ''' This function updates the song's sign

    Keyword argument:

    song_name -- The name of the given song

    artisit_name -- A song object with the updated information of the
    song
    '''
    conn = None
    try:
        conn = connect_to_Lib()
        cur = conn.cursor()

        # Update artist name by the song's title
        cur.execute("UPDATE song SET artist = %s where title = %s",
                    [artist_name, song_name])
        # commit the change and close the cursor
        conn.commit()
        conn.close()
    except (Exception, psycopg2.DatabaseError) as error:
        raise error
    finally:
        if conn is not None:
            conn.close()
    pass


def library_update_title(song_name, updated_name):
    '''This function updates the song's title given the song's
    id in the song table

    Keyword argument:

    song_name -- title of the song to be changed

    updated_name -- The updated song's title
    '''
    conn = None
    try:
        conn = connect_to_Lib()
        cur = conn.cursor()
        # obtain the old name by the id provided
        cur.execute("SELECT FROM song WHERE title = %s", [song_name])
        old_name = cur.fetchone()[0][0]
        # DROP the foreign key constraint in the signature table
        cur.execute("""ALTER TABLE signature DROP CONSTRAINT
                    signature_title_fkey""")
        # Change the title columns in signature table
        cur.execute("UPDATE signature SET title = %s WHERE title = %s",
                    [updated_name, old_name])
        # Change title in song table
        cur.execute("UPDATE song SET title = %s WHERE title = %s",
                    [updated_name, song_name])
        # Add back the foreign key constraint on the signature table
        cur.execute("""ALTER TABLE signature ADD CONSTAINT signature_title_fkey
                    FOREIGN KEY (title) REFERENCES song(title)""")
        # Commit the change and close cursor
        conn.commit()
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        raise error
    finally:
        if conn is not None:
            conn.close()


def library_update_song(song_name, updated_song):
    ''' This function updates every field of the given song,
    including signature

    Keyword argument:

    song_name -- The name of the given song

    updated_song -- The song with updated information of the song,
    a Song object
    '''
    conn = None
    try:
        # Insert song information into song table
        song_update_sql = """UPDATE song
                             SET artist = data.artist
                             SET sampling_rate = data.sampling_rate
                             SET length = data.length
                             FROM (VALUES %s) AS data (title, artist,
                             sampling_rate, length)
                             where song.title = title"""
        cur.execute(song_insertion_sql, [song_name, updated_song.artist,
                                         updated_song.sampling_rate,
                                         updated_song.T])
        # delete song-related signature information in the table
        cur.execute("DELETE * FROM signature where title = %s", song_name)
        # Insert new signature into signature table
        signature_insertion_sql = """INSERT INTO signature(title, window_id,
                                     window_time, signature) VALUES %s"""
        # Convert numpy array to tuples for bulk insertion
        sig_tup = tuple(map(tuple, updated_song.signature))
        num_of_windows = len(updated_song.window_centers)
        sig_ins_dataset = [(song_name, i + 1,
                            updated_song.window_centers[i],
                            "{"+",".join([str(s) for s in sig_tup[i]])+"}")
                           for i in range(num_of_windows)]
        # Bulk insert into signature table
        execute_values(cur, signature_insertion_sql,
                       sig_ins_dataset, template=None)
        conn.commit()
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        raise error
    finally:
        if conn is not None:
            conn.close()


def snippet_analyzer(snippet):
    ''' This function analyzes the snippet signal and find
    the match in the song library

    Keyword Argument:

    snippet_signal -- A dictionary with fields including
    signal, name, sampling_rate, and audio_source
    '''
    snippet_signal = snippet['audio_signal']
    snippet_name = snippet['audio_name']
    sampling_rate = snippet['sampling_rate']
    snippet_address = snippet['audio_source']
    # Construct a Song Object called snippet_song
    snippet_song = Song.Song(snippet_signal, title_info=snippet_name,
                             address_info=snippet_address,
                             sampling_rate_info=sampling_rate)
    # Use the signature of the snippet to find match in the
    # song library
    matched_song, matched_time, params_cp, T, acy_level = signature_match(
        snippet_song.signature)
    # Return the matching information
    print(f"""The snippet matches with the song with title
         {matched_song[1]}, created by {matched_song[2]}""")
    print(f"""The recording has a sampling rate of {int(matched_song[3])}
          and lasts {round(float(matched_song[4]), 1)} seconds""")
    print(f"""The snippet matches with {matched_song[1]}
          at as early as {float(min(matched_time))} seconds""")
    K = params_cp.k
    L = params_cp.l
    lsh_family = str(params_cp.lsh_family)
    print(f"""Use {K} {lsh_family} function and {L} hash tables,
         {T} probes for approximate NN search.""")


def signature_match(snippet_signature):
    ''' This function searches the the matching signature by using
    the approximate nearest neighborhood searches. The searching requries
    the falconn_util, which requires FALCONN-LIB.

    Keyword Argument:

    snippet_signature -- Signature of passed-in snippet. A numpy ndarray.
    '''

    conn = None
    try:
        # Establish connection to the database and cursor for interaction
        # on the database
        conn = connect_to_Lib()
        cur = conn.cursor()

        # Retrieve signature matrix from the signature table
        sig_retrieval_sql = "SELECT signature FROM signature"
        cur.execute(sig_retrieval_sql)
        rows = cur.fetchall()

        # Construc the signature matrix
        sig_mat = np.zeros([len(rows), len(*rows[0])])
        for i in range(len(rows)):
            sig_mat[i, :] = list(*rows[i])

        # Use the signature matrix to construct the falconn_queryable
        falconn_tab = falconn_util.falconn_table(sig_mat)
        # Retrieve the construction parameters for information
        # needed for columns in the falconn table
        params_cp = falconn_tab._params
        falconn_queryable, acy_level = falconn_util.falconn_que(
            falconn_tab, sig_mat)
        T = falconn_queryable.get_num_probes()

        # Use the falconn_queryable to find nearest neighbor of the signature
        nn_rowNum = falconn_util.falconn_search(snippet_signature,
                                                falconn_queryable)
        # Flatten the list
        nn_rowNum = [i for sublist in nn_rowNum for i in sublist]

        # Use the nearest neighbor index to find the corresponding song
        # title and matching time
        rslt = []
        for rowNum in nn_rowNum:
            if rowNum != -1:
                cur.execute("""WITH tmp AS (
                            SELECT title, window_time, ROW_NUMBER() OVER()
                            AS rowNum FROM signature
                            )SELECT title, window_time FROM tmp
                            WHERE rowNum = %s
                            """, [int(rowNum+1)])
                rslt.append(cur.fetchone())

        # Calculate the frequency appearance of the snippet signature
        title, window_time = zip(*rslt)
        title_count = Counter(title)
        print(title_count.most_common(3))
        matched_title = title_count.most_common(1)[0][0]
        matched_time = [window_time[i]
                        for i in range(len(window_time))
                        if title[i] == matched_title]
        # return the information of the matching song and the
        # matched time in the song
        matched_song = library_read(matched_title)[0]
        cur.close()
        return [matched_song, matched_time, params_cp, T, acy_level]
    except (Exception, psycopg2.DatabaseError) as error:
        raise error
    finally:
        if conn is not None:
            conn.close()
