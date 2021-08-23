""" Package Import """
import sys
import numpy as np
import torch
import chess.pgn
import os 
import joblib
import time

from sklearn.preprocessing import StandardScaler
from pickle import load

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
            [list]: material_1, material_2, board_positions_1, board_positions_2, board_maps
        """
        board_maps_1, board_maps_2 = get_board_map(board)
        board_maps = np.concatenate((board_maps_1, board_maps_2))

        return board_maps

    def make_scalar_planar(scalar_value):
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
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    """
    configurations = {
        "Contempt": "type=spin, default=",
        "Analysis Contempt": "type=combo",
        "Threads": ,
        "Hash": ,
        "Clear Hash": ,
        "Ponder": ,
        "MultiPV": ,
        "Skill Level": ,
        "Slow Mover": ,
        "UCI_Chess960": ,
        "UCI_AnalyseMode": ,
        "UCI_LimitStrength": ,
        "UCI_Elo": ,
        "UCI_ShowWDL": ,
        "SyzygyPath": ,
        "SyzygyProbeDepth": ,
        "Syzygy50MoveRule": ,
        "SyzygyProbeLimit": ,
        "Use NNUE": ,
        "EvalFile": ,
    }
    """
    # engine.configure({"UCI_ELO": 2800})
    engine.configure({"Threads": 1})
    engine.configure({"SyzygyPath": "syzygy_tablebases/wdl/"})
    # engine.configure({"Move Overhead": 1000})

    # load standardization/normalization of the model from its training phase
    scaler_input = joblib.load("scalers/Scaler(CTM__2.1__tc60__input).save")
    scaler_prediction = joblib.load("scalers/Scaler(CTM__2.1__tc60__prediction).save")

    # set model version name
    model_version = 'CTM Net 2.1'

    # load CTM Net
    model = torch.load("CTM__v2.1_tc60_epochs9")
    model.to(device)
    model.eval()

    opening_moves = True

    inputs = []
    while True:
        # get actual output from gui to input for engine
        if(inputs):
            command_from_engine = inputs.pop()
        else:
            command_from_engine = input()
        
        # stop game
        if(command_from_engine == 'quit'):
            pass
        
        # question: uci, answer: id name, id author, uciok 
        elif(command_from_engine == 'uci'):
            command('id name CTM Net 1.0 in combination with Stockfish13', out=out)
            command('id author Tillmann Rheude', out=out)
            command('uciok', out=out)

        # question: isready, answer: readyok
        elif(command_from_engine == 'isready'):
            command('readyok', out)
        
        # question: ucinewgame, answer: init the board for a new chess game
        elif(command_from_engine == 'ucinewgame'):
            # get board and create a bitboard out of it
            board = chess.Board()
            opening_moves = True
        
        # question: position, answer:
        elif(command_from_engine.startswith('position')):
            # examples: "position startpos" or "position startpos moves e2e4"
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
                    # choose if white or black player
                    if len(moveslist) % 2 == 0:
                        flip_board = False
                    else:
                        flip_board = True
                    # do the last move from the opponent
                    for move in moveslist[-1:]:
                        san_move = chess.Move.from_uci(move)
                        board.push_san(str(san_move))

                eret_w = [
                    "r1bqk1r1/1p1p1n2/p1n2pN1/2p1b2Q/2P1Pp2/1PN5/PB4PP/R4RK1 w q",
                    "r1n2N1k/2n2K1p/3pp3/5Pp1/b5R1/8/1PPP4/8 w",
                    "r1b1r1k1/1pqn1pbp/p2pp1p1/P7/1n1NPP1Q/2NBBR2/1PP3PP/R6K w",
                    "5b2/p2k1p2/P3pP1p/n2pP1p1/1p1P2P1/1P1KBN2/7P/8 w",
                    "r3kbnr/1b3ppp/pqn5/1pp1P3/3p4/1BN2N2/PP2QPPP/R1BR2K1 w kq",
                    "r2qk2r/ppp1bppp/2n5/3p1b2/3P1Bn1/1QN1P3/PP3P1P/R3KBNR w KQkq",
                    "rnb1kb1r/p4p2/1qp1pn2/1p2N2p/2p1P1p1/2N3B1/PPQ1BPPP/3RK2R w Kkq",
                    "r1b2r1k/ppp2ppp/8/4p3/2BPQ3/P3P1K1/1B3PPP/n3q1NR w",
                    "1nkr1b1r/5p2/1q2p2p/1ppbP1p1/2pP4/2N3B1/1P1QBPPP/R4RK1 w",
                    "1nrq1rk1/p4pp1/bp2pn1p/3p4/2PP1B2/P1PB2N1/4QPPP/1R2R1K1 w",
                    "5k2/1rn2p2/3pb1p1/7p/p3PP2/PnNBK2P/3N2P1/1R6 w",
                    "8/p2p4/r7/1k6/8/pK5Q/P7/b7 w" 
                ]

                eret_b = [
                    "r2r2k1/1p1n1pp1/4pnp1/8/PpBRqP2/1Q2B1P1/1P5P/R5K1 b",
                    "2rq1rk1/pb1n1ppN/4p3/1pb5/3P1Pn1/P1N5/1PQ1B1PP/R1B2RK1 b",
                    "5rk1/pp1b4/4pqp1/2Ppb2p/1P2p3/4Q2P/P3BPP1/1R3R1K b",
                ]

                startposition_fen = ["rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
                                "rnbqkbnr/pp1ppppp/8/8/3pP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 3",
                ]

                flip_board = False

                if flip_board:
                    board = board.mirror()

                for fen in eret_w: 
                    print("FEN: ", fen)
                    board = chess.Board(fen)

                    # count material for player
                    mat_p1, _ = getPlayerMaterial(board)
                    # print(mat_p1)
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
                        prediction = prediction[0][0] # * 1.15

                    print(prediction)
            

        # question: go, answer: 
        elif command_from_engine.startswith('go'):
            # example: "go wtime 270191 btime 268977 movestogo 36" or "go wtime 300000 btime 300000 movestogo 40"
            # get remaining times for both players
            commands = command_from_engine.split()
            idx = commands.index('wtime')

            if(idx >= 0): 
                rem_time_W = int(commands[idx + 1]) / 1000
                idx = commands.index('btime')
                rem_time_B = int(commands[idx + 1]) / 1000 

                if flip_board: 
                    rem_time_p1 = rem_time_B 
                    #rem_time_p2 = rem_time_W
                else: 
                    rem_time_p1 = rem_time_W
                    #rem_time_p2 = rem_time_B

            # multiply prediction with remaining time as prediction is a percent value
            # optional: multiply the resulting time with a factor 
            #if rem_time_p1 < 12 and rem_time_p1 > rem_time_p2:
            #   prediction = (prediction * 0.9) * rem_time_p1
            #else: 
            
            prediction *= rem_time_p1
            # 0,6 oder 0,7

            
            # prediction *= rem_time_p1
            # send prediction to engine
            print("The engine will think about ", prediction, " seconds right now.")
            print(board)
            bestmove = str((engine.play(board, chess.engine.Limit(time = prediction))).move)

            # push the best move on the current board
            #san_move = chess.Move.from_uci(bestmove)
            #board.push_san(str(san_move))

            # send output
            command("bestmove " + bestmove, out = out)

if __name__ == '__main__':
    main()
