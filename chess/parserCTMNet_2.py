""" Package Import """
import sys
import numpy as np
import torch
import chess.pgn
import os 
import joblib

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

def main():
    """ Helper Functions """
    def getPlayerMaterial(board):
        """Counts the number of materials on the current board for each player (A and B).

        Args:
            board (pgn board): Current board

        Returns:
            [int]: materialPlayerA
            [int]: materialPlayerB
        """
        information = str(board).replace("\n", " ")
        materialPlayer_B = information.count('r') + information.count('n') + information.count('b') + information.count('q') + information.count('k') + information.count('p')
        materialPlayer_W = information.count('R') + information.count('N') + information.count('B') + information.count('Q') + information.count('K') + information.count('P')
        return materialPlayer_W, materialPlayer_B

    def get_board_map(board):
        """ Convert a board to a numpy board map with zeros and ones.

        Args:
            board (chessboard): Chess board 

        Returns:
            [np.ndarray]: Two numpy ndarrays each (6,8,8) for the one-hot-encoded maps

        Example: Board map for the starting position of the white player: 
            [[[0. 0. 0. 0. 0. 0. 0. 0.]
            [0. 0. 0. 0. 0. 0. 0. 0.]
            [0. 0. 0. 0. 0. 0. 0. 0.]
            [0. 0. 0. 0. 0. 0. 0. 0.]
            [0. 0. 0. 0. 0. 0. 0. 0.]
            [0. 0. 0. 0. 0. 0. 0. 0.]
            [1. 1. 1. 1. 1. 1. 1. 1.]
            [0. 0. 0. 0. 0. 0. 0. 0.]]

            [[0. 0. 0. 0. 0. 0. 0. 0.]
            [0. 0. 0. 0. 0. 0. 0. 0.]
            [0. 0. 0. 0. 0. 0. 0. 0.]
            [0. 0. 0. 0. 0. 0. 0. 0.]
            [0. 0. 0. 0. 0. 0. 0. 0.]
            [0. 0. 0. 0. 0. 0. 0. 0.]
            [0. 0. 0. 0. 0. 0. 0. 0.]
            [0. 0. 0. 0. 1. 0. 0. 0.]]

            [[0. 0. 0. 0. 0. 0. 0. 0.]
            [0. 0. 0. 0. 0. 0. 0. 0.]
            [0. 0. 0. 0. 0. 0. 0. 0.]
            [0. 0. 0. 0. 0. 0. 0. 0.]
            [0. 0. 0. 0. 0. 0. 0. 0.]
            [0. 0. 0. 0. 0. 0. 0. 0.]
            [0. 0. 0. 0. 0. 0. 0. 0.]
            [0. 0. 0. 1. 0. 0. 0. 0.]]

            [[0. 0. 0. 0. 0. 0. 0. 0.]
            [0. 0. 0. 0. 0. 0. 0. 0.]
            [0. 0. 0. 0. 0. 0. 0. 0.]
            [0. 0. 0. 0. 0. 0. 0. 0.]
            [0. 0. 0. 0. 0. 0. 0. 0.]
            [0. 0. 0. 0. 0. 0. 0. 0.]
            [0. 0. 0. 0. 0. 0. 0. 0.]
            [0. 0. 1. 0. 0. 1. 0. 0.]]

            [[0. 0. 0. 0. 0. 0. 0. 0.]
            [0. 0. 0. 0. 0. 0. 0. 0.]
            [0. 0. 0. 0. 0. 0. 0. 0.]
            [0. 0. 0. 0. 0. 0. 0. 0.]
            [0. 0. 0. 0. 0. 0. 0. 0.]
            [0. 0. 0. 0. 0. 0. 0. 0.]
            [0. 0. 0. 0. 0. 0. 0. 0.]
            [0. 1. 0. 0. 0. 0. 1. 0.]]

            [[0. 0. 0. 0. 0. 0. 0. 0.]
            [0. 0. 0. 0. 0. 0. 0. 0.]
            [0. 0. 0. 0. 0. 0. 0. 0.]
            [0. 0. 0. 0. 0. 0. 0. 0.]
            [0. 0. 0. 0. 0. 0. 0. 0.]
            [0. 0. 0. 0. 0. 0. 0. 0.]
            [0. 0. 0. 0. 0. 0. 0. 0.]
            [1. 0. 0. 0. 0. 0. 0. 1.]]]
        """
        # init ids and numeric synonyms for them
        ids = ['r', 'n', 'b', 'q', 'k', 'p', 'P', 'K', 'Q', 'B', 'N', 'R']
        ids_numeric = np.arange(1,13)

        # convert board to list to iterate in a nicer way later
        board = str(board).replace("\n", " ").split()

        np_board = np.zeros((64))

        for i, figure in enumerate(board):
            for letter in ids: 
                if letter == ".":
                    continue
                elif letter == figure:
                    np_board[i] = ids.index(letter) + 1
                    continue

        np_board = np.reshape(np_board, (8,8))

        # init numpy arrays for white figures and black figures
        board_maps_W = np.zeros((6,8,8))
        board_maps_B = np.zeros((6,8,8))

        for i in range(board_maps_W.shape[0]):
            board_maps_W[i] = np_board
            board_maps_B[i] = np_board

        # decode boards to 0 and 1 resp. for one (or no) figure on the board
        for i, letter in enumerate(ids_numeric[:6]):
            board_maps_B[i] = board_maps_B[i] == letter
            board_maps_B[i].astype(int)

        for i, letter in enumerate(ids_numeric[6:]):
            board_maps_W[i] = board_maps_W[i] == letter
            board_maps_W[i].astype(int)

        # return both board maps
        return board_maps_W, board_maps_B

    def get_both_board_maps(board):
        """ Extracts multiple information about moves done in a game.

        Args:
            game ([pgngame]): A chess game
            board_flip ([boolean]): True or false resp. who won the game (True = Black won, False = White won)

        Returns:
            [np array]: board_maps
        """
        board_maps_1, board_maps_2 = get_board_map(board)
        board_maps = np.concatenate((board_maps_1, board_maps_2))

        return board_maps

    def make_scalar_planar(scalar_value):
        """ Helper function to make a scalar value planar.

        Args:
            scalar_value ([int]): Any scalar value.

        Returns:
            [np array]: numpy array filled with the scalar value in shape (1,8,8)
        """
        plane = np.zeros((1,8,8))
        plane.fill(scalar_value)
        return plane

    class Outstream(object):
        """ Helper class for communicating with CuteChess

        Args:
            object (object): sys.stdout
        """
        def __init__(self, stream):
            self.stream = stream
        def write(self, written_data):
            self.stream.write(written_data)
            self.stream.flush()
            sys.stderr.write(written_data)
            sys.stderr.flush()
        def __getattr__(self, attr):
            return getattr(self.stream, attr)

    def command(line, out):
        print(line, file = out)

    # init outstream
    out = Outstream(sys.stdout)
    # choose device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # choose engine
    engine_name = 'Stockfish13'
    engine_path = "/usr/src/app/chess/cutechess-cli/engines/stockfish_13_linux_x64_bmi2"

    #engine_name = 'Stockfish14'
    #engine_path = "/usr/src/app/chess/cutechess-cli/engines/stockfish_14_x64_bmi2"

    #engine_name = 'Igel'
    #engine_path = "/usr/src/app/chess/cutechess-cli/engines/igel-x64_bmi2"

    #engine_name = 'Ethereal'
    #engine_path = "/usr/src/app/chess/cutechess-cli/engines/Ethereal_12_75"

    #engine_name = 'Xiphos'
    #engine_path = "/usr/src/app/chess/cutechess-cli/engines/xiphos-sse"

    #engine_name = 'Slow Chess Blitz'
    #engine_path = "/usr/src/app/chess/cutechess-cli/engines/slow64_linux_sse"

    #engine_name = 'RubiChess1.9'
    #engine_path = "/usr/src/app/chess/cutechess-cli/engines/RubiChess_1_9"

    #engine_name = 'Wasp450'
    #engine_path = "/usr/src/app/chess/cutechess-cli/engines/Wasp450-linux"

    engine = chess.engine.SimpleEngine.popen_uci(engine_path)

    # engine.configure({"UCI_Elo": 2850})
    engine.configure({"Threads": 1})
    engine.configure({"SyzygyPath": "syzygy_tablebases/wdl/"})

    ### To print options:
    #print(engine.options["Contempt"])
    #print(engine.options["Analysis Contempt"])
    #print(engine.options["Threads"])
    #print(engine.options["Hash"])
    #print(engine.options["Clear Hash"])
    #print(engine.options["Ponder"])
    #print(engine.options["MultiPV"])
    #print(engine.options["Skill Level"])
    #print(engine.options["Slow Mover"])
    #print(engine.options["UCI_AnalyseMode"])
    #print(engine.options["UCI_LimitStrength"])
    #print(engine.options["UCI_Elo"])
    #print(engine.options["UCI_ShowWDL"])
    #print(engine.options["SyzygyPath"])
    #print(engine.options["SyzygyProbeDepth"])
    #print(engine.options["Syzygy50MoveRule"])
    #print(engine.options["EvalFile"])
    #print(engine.options["SyzygyProbeLimit"])
    #print(engine.options["Use NNUE"])

    # load standardization/normalization of the model from its training phase
    scaler_input = joblib.load("scalers/Scaler(CTM__2.1__tc60__input).save")
    scaler_prediction = joblib.load("scalers/Scaler(CTM__2.1__tc60__prediction).save")

    # set model version name
    model_version = 'CTM Net 2.1'

    # load CTM Net
    model = torch.load("CTM__v2.1_tc60_epochs9")
    model.to(device)
    model.eval()

    # define if opening suites are used or not 
    opening_moves = True
    first_move = True
    move_count = 0

    inputs = []
    while True:
        # get actual output from gui to input for engine
        if(inputs):
            command_from_engine = inputs.pop()
        else:
            command_from_engine = input()
        
        if(command_from_engine == 'quit'):
            pass
        
        elif(command_from_engine == 'uci'):
            command('id name CTM Net 2.1 in combination with ' + engine_name, out=out)
            command('id author Tillmann Rheude', out=out)
            command('uciok', out=out)

        elif(command_from_engine == 'isready'):
            command('readyok', out)
        
        elif(command_from_engine == 'ucinewgame'):
            # get board and create a bitboard out of it
            board = chess.Board()
            opening_moves = True
            first_move = True
            engine = chess.engine.SimpleEngine.popen_uci(engine_path)
            engine.configure({"Threads": 1})
            engine.configure({"SyzygyPath": "syzygy_tablebases/wdl/"})
            move_count = 0
        
        elif(command_from_engine.startswith('position')):
            # examples: "position startpos" or "position startpos moves e2e4 d7d6"

            idx = command_from_engine.find('moves')
            if(idx >= 0): 
                moveslist = command_from_engine[idx:].split()[1:]
            else: # if idx = -1 there is no "moves" in command_from_engine
                moveslist = []

            if(opening_moves == False and moveslist == []):
                # get materials for both players 
                mat_p1 = 16
                # mat_p2 = 16

                # declare that engine is white player as the engine starts with the first move
                flip_board = False

                # get board 
                board = chess.Board()

                # get player1 material
                mat_p1 = np.array(mat_p1).reshape(-1,1)
                # standardize/normalize material with scaler from training process
                mat_p1 = scaler_input.transform(mat_p1)[0][0]
                # make material planar
                mat_p1 = make_scalar_planar(mat_p1)[np.newaxis, :]

                # get board and create a boardmap out of it
                board_map = get_both_board_maps(board)[np.newaxis, :]
                
                # stack features together as input for CTM Net
                features = np.hstack((board_map, mat_p1))

            else: 
                if opening_moves:
                    # init a new board
                    board = chess.Board()

                    # choose if white or black player
                    if len(moveslist) % 2 == 0:
                        flip_board = False
                    else:
                        flip_board = True
                    
                    # do opening moves on the board
                    for move in moveslist:
                        san_move = chess.Move.from_uci(move)
                        board.push_san(str(san_move))
                    
                    # set opening moves to false as we did them right now
                    opening_moves = False
                else: 
                    if first_move == False:
                        # push the best move on the current board
                        san_move = chess.Move.from_uci(bestmove)
                        board.push_san(str(san_move))

                    # choose if white or black player
                    if len(moveslist) % 2 == 0:
                        flip_board = False
                    else:
                        flip_board = True
                    # do the last move from the opponent
                    for move in moveslist[-1:]:
                        san_move = chess.Move.from_uci(move)
                        board.push_san(str(san_move))

                if flip_board:
                    board = board.mirror()
                    
                # count material for player
                mat_p1, _ = getPlayerMaterial(board)
                mat_p1 = np.array(mat_p1).reshape(-1,1)
                # standardize/normalize material with scaler from training process
                mat_p1 = scaler_input.transform(mat_p1)[0][0]
                # make material planar
                mat_p1 = make_scalar_planar(mat_p1)[np.newaxis, :]

                # get board and create a boardmap out of it
                board_map = get_both_board_maps(board)[np.newaxis, :]

                # stack features together as input for CTM Net
                features = np.hstack((board_map, mat_p1))
            
                if flip_board:
                    board = board.mirror()

            # features to tensor
            x = torch.tensor(features, dtype = torch.float).to(device)
            
            # here comes CTM Net to predict the perfect time y to think about this position from input x
            with torch.no_grad():
                # de-normalize prediction with min, max (mean, ...) from training process
                prediction = scaler_prediction.inverse_transform(np.array([model(x).item()]).reshape(-1,1))
                prediction = round(prediction[0][0] * 1, 4) / 1000
            
        # question: go, answer: 
        elif command_from_engine.startswith('go'):
            # example: "go wtime 270191 btime 268977" or "go wtime 300000 btime 300000"
            # get remaining times for both players

            commands = command_from_engine.split()

            if flip_board: 
                idx = commands.index('btime')
                rem_time_B = int(commands[idx + 1]) 
                rem_time_p1 = rem_time_B 
                idx = commands.index('wtime')
                rem_time_W = int(commands[idx + 1]) 
                rem_time_p2 = rem_time_W
            else: 
                idx = commands.index('wtime')
                rem_time_W = int(commands[idx + 1]) 
                rem_time_p1 = rem_time_W 
                idx = commands.index('btime')
                rem_time_B = int(commands[idx + 1]) 
                rem_time_p2 = rem_time_B 

            #if prediction > 0.04: 
            #    prediction = prediction * (1 + (prediction * 2))

            prediction *= rem_time_p1
            
            #if prediction > 1.2: prediction = prediction * 0.7
            #if prediction < 0.3: prediction = prediction * 1.3

            # searching without prediction:
            # bestmove = str((engine.play(board, chess.engine.Limit(white_clock = rem_time_W/1000, black_clock = rem_time_B/1000))).move)

            # send prediction to engine
            bestmove = str((engine.play(board, chess.engine.Limit(time = prediction))).move)
            #print(bestmove)

            move_count += 1
            first_move = False

            # send output
            command("bestmove " + bestmove, out = out)

if __name__ == '__main__':
    main()
