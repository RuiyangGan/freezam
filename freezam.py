# Import required packages and Song class
import Song
import argparse
import os
import IO
import DbManager
import logging
import logging.config
from logging.handlers import RotatingFileHandler


# Add the main parser freezam
freezam = argparse.ArgumentParser(prog="freezam")
subzams = freezam.add_subparsers(help="sub-command-help",
                                 dest="subparser_name")

# Create the parser for the add command
freezam_add = subzams.add_parser('add', help='add help',
                                 description='Add the song to the library')
freezam_add.add_argument('song', type=str, help='song help')
freezam_add.add_argument('--title', nargs='?', default=None,
                         type=str, help='title help')
freezam_add.add_argument('--artist', nargs='?', default=None,
                         type=str, help='artist help')
freezam_add.add_argument('--verbose', '-v',
                         help='verbose help', action="store_true")

# Create the parser for the delete command
freezam_delete = subzams.add_parser('delete', help='delete help',
                                    description="""Delete the song
                                                 in the library""")
freezam_delete.add_argument('song', type=str, help='song help')
freezam_delete.add_argument('--verbose', '-v',
                            help='verbose help', action="store_true")

# Create the parser for the identify command
freezam_identify = subzams.add_parser('identify', help='identify help',
                                      description='identify the given snippet')
freezam_identify.add_argument('snippet', type=str, help='snippet help')
freezam_identify.add_argument('--verbose', '-v',
                              help='verbose help', action="store_true")

# Create the parser for the list_song command
freezam_list_song = subzams.add_parser('list_song', help="list_song help",
                                       description='List song in library')
# Create the parser for verbose option
freezam_list_song.add_argument('--verbose', '-v',
                               help='verbose help', action="store_true")

# Create the parser for the update command
freezam_update = subzams.add_parser('update', help='update help',
                                    description="""Update information
                                                   about the song""")
freezam_update.add_argument('song_title', type=str, help='song_title help')
freezam_update.add_argument('--title', type=str, nargs='?',
                            help='updated-title help')
freezam_update.add_argument('--artist', type=str, nargs='?',
                            help='updated-artist help')
freezam_update.add_argument('--song', type=str, nargs='?',
                            help='song help')

# Create the parser for verbose option
freezam_update.add_argument('--verbose', '-v',
                            help='verbose help', action="store_true")

# Create logger for file Handler, requires logging.conf
logging.config.fileConfig("./logging.conf")

# define a Handler which writes INFO messages or higher
# to stdout
sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
# Create a simple formatter for stdout
shformatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
# tell the handler to use this format
sh.setFormatter(shformatter)


# Write the function for adding, listing, and identifying
# song in library
def Add(song, title=None, artist_info=None):
    song = IO.audio_to_signal(song)
    if title is None:
        title = song['audio_name']
    song = Song.Song(song['audio_signal'],
                     sampling_rate_info=song['sampling_rate'],
                     address_info=song['audio_source'],
                     title_info=title,
                     artist_info=artist_info)
    DbManager.library_add(song)


def Add_dir(dir):
    song_list = [os.path.join(dir, p) for p in os.listdir(dir)]
    for song in song_list:
        Add(song)


def List():
    rows = DbManager.library_read()
    for row in rows:
        print(f"""id:{row[0]}, title:{row[1]}, Artist:{row[2]},
                Sampling rate:{int(row[3])},
                Length:{round(float(row[4]), 1)} seconds""")
    print(f"""Total of {len(rows)} song in the Library""")


def Identify(snippet):
    snippet = IO.audio_to_signal(snippet)
    DbManager.snippet_analyzer(snippet)


def Delete(song):
    DbManager.library_delete(song)


def Update(song_title, title, artist, song):
    if song is not None:
        DbManager.library_update_song(song_title, song)
    if artist is not None:
        DbManager.library_update_artist(song_title, artist)
    if title is not None:
        DbManager.library_update_title(song_title, title)


if __name__ == '__main__':
    # Create the main logger
    logger = logging.getLogger(__name__)
    # Capture other warnings and redirect to the log file
    logging.captureWarnings(True)
    warnings_logger = logging.getLogger("py.warnings")
    warnings_file_handler = RotatingFileHandler(u'freezam.log')
    warnings_logger.addHandler(warnings_file_handler)
    args = freezam.parse_args()
    if args.verbose:
        # Add a stream Handler to the logger and print to stdout
        # add the handler to the root logger
        logger.addHandler(sh)
    if args.subparser_name == 'add':
        # Determine whether the given path is a file for a directory
        if os.path.isdir(args.song):
            Add_dir(args.song)
            logger.info(f"""Add the files in the directory
                            {args.song} to the library""")
        else:
            Add(args.song, args.title, args.artist)
            logger.info(f"Add the song {args.song} to the library")
    elif args.subparser_name == 'identify':
        Identify(args.snippet)
        logger.info(f"Identify the snippet {args.snippet}")
    elif args.subparser_name == 'list_song':
        List()
        logger.info("List all songs in the library")
    elif args.subparser_name == 'delete':
        Delete(args.song)
        logger.info(f"Delete the song {args.song} successfully")
    elif args.subparser_name == 'update':
        Update(args.song_title, args.title, args.artist, args.song)
        logger.info(f"Updates the song {args.song_title} successfully")
