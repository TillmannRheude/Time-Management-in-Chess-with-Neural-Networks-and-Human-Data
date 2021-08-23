""" Package Import """
import subprocess
import os
from rtpt import RTPT

# create RTPT
rtpt = RTPT(name_initials = "TR", experiment_name = "Preprocessing", max_iterations = 1)
rtpt.start()

# set directory
path_to_datasets = "/usr/src/app/datasets/lichess"

# select filter file which contains the filter criteria
filterfile = "/usr/src/app/datasets/lichess/filtertags.txt"

# sanity check if pgn-extract is found
command = "pgn-extract --version"
result = subprocess.run(command.split(' '), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout = result.stdout.decode("UTF-8")
stderr = result.stderr.decode("UTF-8")
print(stdout, stderr)
print("-" * 30)

print("Extraction starts")
print("-" * 30)

# files to extract
missing_filenames = [
            "lichess_db_standard_rated_2020-08.pgn",
            "lichess_db_standard_rated_2020-07.pgn",
            "lichess_db_standard_rated_2020-04.pgn",
            "lichess_db_standard_rated_2020-03.pgn",
            "lichess_db_standard_rated_2020-02.pgn",
            "lichess_db_standard_rated_2020-01.pgn"
            ]

# start extracting
for subdir, dirs, files in os.walk(path_to_datasets):
    for filename in files:
        if filename.endswith(".pgn"):
            if filename in missing_filenames:
                # print actual file to extract
                print("Actually we are here: ", os.path.join(subdir, filename))

                # png-extract command 
                command = 'pgn-extract -t ' + filterfile + ' -M -o datasets/' + filename + ' ' + os.path.join(subdir, filename)
                result = subprocess.run(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout = result.stdout.decode('utf-8')
                stderr = result.stderr.decode('utf-8')
                print(stdout, stderr)
                print("-" * 30)
