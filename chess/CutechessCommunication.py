""" Package Import """
import subprocess
import time

def run_command(command):  
    result = subprocess.run(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout = result.stdout.decode('utf-8')
    stderr = result.stderr.decode('utf-8')
    print(stdout, stderr)
    return stdout, stderr

def run_comma_command(command):
    # invoke process
    result = subprocess.Popen(command.split(","), shell=False, stdout=subprocess.PIPE)

    # Poll process.stdout to show stdout live
    while True:
        output = result.stdout.readline()
        if result.poll() is not None:
            break
        if output:
            print(output.strip())
        rc = result.poll()
    
    """
    result = subprocess.run(command.split(","), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout = result.stdout.decode('utf-8')
    stderr = result.stderr.decode('utf-8')
    print(stdout, stderr)
    return stdout, stderr
    """

# Define Engine 1
engine_1_name = "CTM Net 2.1"
cmd_e1 = "python3 parserCTMNet_2.py"

# Define Engine 2
engine_2_name = "SF13"
cmd_e2 = "./engines/stockfish_13_linux_x64_bmi2"

#engine_2_name = "SF14"
#cmd_e2 = "./engines/stockfish_14_x64_bmi2"

#engine_2_name = "Igel"
#cmd_e2 = "./engines/igel-x64_bmi2"

#engine_2_name = "Ethereal"
#cmd_e2 = "./engines/Ethereal_12_75"

#engine_2_name = "Xiphos"
#cmd_e2 = "./engines/xiphos-sse"

#engine_2_name = "Slow Chess Blitz"
#cmd_e2 = "./engines/slow64_linux_sse"

#engine_2_name = "RubiChess1.9"
#cmd_e2 = "./engines/RubiChess_1_9"

# ... or any other uci-ready chess engine.

# Set directory of CuteChess
directory = "chess/cutechess-cli/"

# Set TimeControl for the games
timecontrol = "0/0:60+0"  # "0/5:00+0"

# Set save path for the pgn-file
save_pgn_path = "chess/game_results/vs_SF13_60+0_midStartOfCTMNet4.2.pgn"

# Set chess variant
variant = "standard"

# Set number of games which should be played
n_games = str(100)
rounds = str(1)
repeat = str(2)

# Set Opening Suite
# opening_file = "chess_engines/stockfish_13_linux_x64_bmi2/cutechess-cli/UHO_V3_6mvs_+090_+099.pgn"
opening_file = "chess/cutechess-cli/Pohl_AntiDraw_Openings_V1.5/Final/UHO_V3_8mvs_+150_+159.pgn"

# Sanity check if Cutechess is accessible
command = "cutechess-cli --version"
stdout, stderr = run_command(command)

# time.sleep(600000) # for debug & testing
# Run cutechess
command = "cutechess-cli,-variant," + variant + ",-openings,file=" + opening_file + ",-pgnout," + save_pgn_path + ",-resign,movecount=5,score=600,-draw,movenumber=30,movecount=4,score=20,-engine,name=" + engine_1_name + ",cmd=" + cmd_e1 + ",-engine,name=" + engine_2_name + ",option.Threads=1,option.SyzygyPath=syzygy_tablebases/wdl/,cmd=" + cmd_e2 + ",-each,dir=" + directory + ",proto=uci,tc=" + timecontrol + ",-games," + n_games + ",-rounds," + rounds + ",-repeat," + repeat + ",-debug"
stdout, stderr = run_comma_command(command)

# other options for CuteChess
# ,-concurrency,8 ,option.UCI_Elo=2850,
#-resign movecount=5 score=600 -draw movenumber=30 movecount=4 score=20

