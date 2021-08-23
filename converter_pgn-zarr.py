""" File Import """
import chess.pgn
import zarr

import numpy as np


""" Functions """
def import_multiple_games(pgn_file, maxGames, type = "Lichess Standard"):
    """Import multiple games from a pgn file

    Args:
        pgn_file ([open(path)]): opened path to the pgn_file

    Returns:
        [list]: Multiple games
    """
    # create list which contains all games
    all_games = []

    while True:
        # read one game from pgn_file
        game = chess.pgn.read_game(pgn_file)

        # if no game is left anymore or a limit of 100.000 games is reached: stop importing
        if (game is None) or (len(all_games) == maxGames):
            break

        # safe all games in list all_games
        all_games.append(game)
        
        # print actual status of imported games
        if ((len(all_games) % 10000) == 0) and ((len(all_games)) != 0): 
            print("Games imported: ", len(all_games))
    
    return all_games

def get_result_OHE(result):
    """One-Hot encoding for the result

    Args:
        result (String): Result in form '1-0', '1/2-1/2' or '0-1'

    Returns:
        [int]: alternative one hot encoding of the result
        [np.array (3,)]: One-Hot encoded result in form (1,0,0), (0,1,0) or (0,0,1) respectively
    """
    one_hot_encoded_result = np.zeros(3)
    alternative_ohe_result = 0
    if(result == "1-0"):
        one_hot_encoded_result[0] = 1
        alternative_ohe_result = 0
    if(result == "1/2-1/2"):
        one_hot_encoded_result[1] = 1
        alternative_ohe_result = 1
    if(result == "0-1"):
        one_hot_encoded_result[2] = 1
        alternative_ohe_result = 2
    return alternative_ohe_result #, one_hot_encoded_result

def get_sec(time):
    """ Convert given time into seconds

    Args:
        time ([str]): given time

    Returns:
        [int]: seconds regarding to the given time
    """
    m, s = time[-4:-2], time[-2:]
    if time[-4:2] == '' : m = 0
    if time[-2:] == '' : s = 0

    return int(m) * 60 + int(s)

def get_player_material(board):
    """Counts the number of materials on the current board for each player (A and B).

    Args:
        board (pgn board): Current board

    Returns:
        [int]: materialPlayerA
        [int]: materialPlayerB
    """
    # clear board to read out information
    information = str(board).replace("\n", " ")

    # get black and white material
    materialPlayer_B = information.count('r') + information.count('n') + information.count('b') + information.count('q') + information.count('k') + information.count('p')
    materialPlayer_W = information.count('R') + information.count('N') + information.count('B') + information.count('Q') + information.count('K') + information.count('P')
    
    # return materials for each player
    return materialPlayer_W, materialPlayer_B

def get_piece_OHE(piece):
    """One-Hot encode a given piece into a vector

    Args:
        piece ([String]): Piece (e.g. r or R)

    Returns:
        [np.array]: (12,) one-hot encoded vector for the piece
    """
    ids = ['r', 'n', 'b', 'q', 'k', 'p', 'P', 'K', 'Q', 'B', 'N', 'R']
    one_hot_encoded_piece = np.zeros(len(ids))
    if(piece in ids):
        one_hot_encoded_piece[ids.index(piece)] = 1
    return one_hot_encoded_piece

def get_board_positions_pieces_OHE(board):
    """Convert a given board into many one-hot encoded vectors for each position on the board

    Args:
        board ([board]): The current board

    Returns:
        [np.array]: (8,8,12) matrix with positions as one-hot encoded vectors
                    8 rows (chess board up to down)
                    8 columns (chess board left to right)
                    12 possible pieces on every row/column

    Example:
        The start position of the black pieces front row:
        [0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
        [0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
        [0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
        [0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
        [0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
        [0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
        [0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
        [0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
        The start position of the black pieces back row:
        [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
        [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
        [0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
        [0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
        [0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
        [0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
        [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
        [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
    """
    information = str(board).replace("\n", " ")
    one_hot_encoded_board_positions = np.zeros(shape = (8,8,12))
    x = 0
    y = 0
    for piece in information:
        if(piece == " "):
            continue
        one_hot_encoded_board_positions[y, x, :] = get_piece_OHE(piece)
        if(x <= 7):
            x += 1 
        if(x > 7):
            x = 0
            y += 1
    return one_hot_encoded_board_positions

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

def extract_move_information(game, board_flip):
    """ Extracts multiple information about moves done in a game.

    Args:
        game ([pgngame]): A chess game
        board_flip ([boolean]): True or false resp. who won the game (True = Black won, False = White won)

    Returns:
        [list]: material_1, material_2, board_positions_1, board_positions_2, board_maps
    """
    # init arrays for return
    material_1, material_2 = [], []
    board_maps = []
    board_positions_1 = []
    board_positions_2 = []

    # get current board
    board = game.board()

    lastmove_is_even = True
    # do moves in the current board
    for index, move in enumerate(game.mainline_moves()):
        if board_flip:
            board = board.mirror()

            if index % 2 != 0 : 
                # helper variable, see below
                lastmove_is_even = False

                # get materials from the first move
                material_1.append(get_player_material(board)[0])

                # get board map for convolutions
                board_maps_1, board_maps_2 = get_board_map(board)
                board_maps_WB = np.concatenate((board_maps_1, board_maps_2))
                board_maps.append(board_maps_WB)
                
                # get board pieces and positions
                board_positions_1.append(get_board_positions_pieces_OHE(board))

            if index % 2 == 0 : 
                # helper variable, see below
                lastmove_is_even = True

                # get materials from the first move
                material_2.append(get_player_material(board)[1])

                # get board pieces and positions
                board_positions_2.append(get_board_positions_pieces_OHE(board))
            
            # mirror board to be able to push next move
            board = board.mirror()
        else: 
            if index % 2 == 0 : 
                # helper variable, see below
                lastmove_is_even = False

                # get materials from the first move
                material_1.append(get_player_material(board)[0])

                # get board map for convolutions
                board_maps_1, board_maps_2 = get_board_map(board)
                board_maps_WB = np.concatenate((board_maps_1, board_maps_2))
                board_maps.append(board_maps_WB)
                
                # get board pieces and positions
                board_positions_1.append(get_board_positions_pieces_OHE(board))

            if index % 2 != 0 : 
                # helper variable, see below
                lastmove_is_even = True

                # get materials from the first move
                material_2.append(get_player_material(board)[1])

                # get board pieces and positions
                board_positions_2.append(get_board_positions_pieces_OHE(board))

        # push to the next move
        board.push(move)
    
    if lastmove_is_even:
        if board_flip:
            board = board.mirror()
        # get materials from the first move
        material_2.append(get_player_material(board)[1])

    return material_1, material_2, board_positions_1, board_positions_2, board_maps

def extract_time_from_comment(commentList, timecontrol):
    """ Extracts the time from a pgn file. The pgn file contains time contents in additional form. Time is
        in brackets behind the moves.

    Args:
        commentList (list): List with comments of a game
        timecontrol (int): time control of the pgn files (no increment supported actually)

    Returns:
        [list]: Multiple time information as lists (times_p1, times_p2, remaining_times_p1, remaining_times_p2)
    """
    # get number of comments in the current board
    n = len(commentList)
    
    # remove backslashes from comments
    for i, cell in enumerate(commentList):
        commentList[i] = cell.replace("\n", " ")

    # safe seconds from comments
    timecontrol = timecontrol # int(game.headers["TimeControl"].split("+")[0])

    # set start times for white and black player
    start_time_p1, start_time_p2 = timecontrol, timecontrol

    # times W
    times_p1 = []
    remaining_times_p1 = []
    # times B
    times_p2 = []
    remaining_times_p2 = []

    for i in range(0, n, 2):
        if(commentList[i] == ''):
            print("There are no comments for this move of the game.")
        else:
            # White times
            numbers_in_comment = [int(s) for s in commentList[i] if s.isdigit()]
            string_clock = [str(i) for i in numbers_in_comment]
            clock = ",".join(string_clock).replace(",", "")
            clock = clock[-5:]

            # if the stamps stands for time left
            time_p1 = start_time_p1 - get_sec(clock)
            times_p1.append(time_p1)
            remaining_times_p1.append(start_time_p1)
            start_time_p1 = start_time_p1 - (start_time_p1 - get_sec(clock))
            
            # Black times
            # print(numbers_in_comment)
            if(i+1 < len(commentList)):
                numbers_in_comment2 = [int(s) for s in commentList[i+1] if s.isdigit()]
                string_clock2 = [str(i) for i in numbers_in_comment2]
                clock2 = ",".join(string_clock2).replace(",", "")
                clock2 = clock2[-5:]

                # if the stamps stands for time left
                time_p2 = start_time_p2 - get_sec(clock2) 
                times_p2.append(time_p2)
                remaining_times_p2.append(start_time_p2)
                start_time_p2 = start_time_p2 - (start_time_p2 - get_sec(clock2))
                
    return times_p1, times_p2, remaining_times_p1, remaining_times_p2

def read_pgn(filepath, maxGames = 100000, timecontrol = 300, type = "Lichess Standard"):
    """ Read in pgn files as numpy arrays.

    Args:
        filepath ([string]): Path to the pgn file
        maxGames ([int]): Maximum value of imported games
        timecontrol ([int]): time control for the imported data (without increment)
        type ([string]): Actually only Lichess Standard supported.

    Returns:
        [np array]: prediction_variable, input_variables, board_maps
    """
    # import pgn file
    pgn_file = open(filepath)
    
    # read all games from pgn_file
    all_games = import_multiple_games(pgn_file, maxGames = maxGames)
    
    # create lists for saving variables
    board_maps = []
    times_p1, remaining_times_p1 = [], []
    materials_p1, materials_p2 = [], []
    board_positions = []
    results = []

    # Counter variables for Summary
    valid_time_counter = 0
    valid_rem_time_counter = 0
    valid_outlier_counter = 0

    # Define end of iteration 
    end = len(all_games)

    # Print checkpoint
    print(len(all_games), " Games will now be read in as arrays")

    # Set type of the game
    if(type == "Lichess Standard"):
        # iterate through every game
        for current_game in range(0, end):
            # if current_game == 1: continue

            # get result
            result = get_result_OHE(all_games[current_game].headers["Result"])

            if result == 2: 
                board_flip = True
            elif result == 0: 
                board_flip = False 
            elif result == 1: 
                continue

            # create arrays for the current game
            curr_times_p1, curr_remaining_times_p1 = [], []
            
            # do moves
            curr_materials_p1, curr_materials_p2, curr_board_positions_p1, _, curr_board_maps = extract_move_information(all_games[current_game], board_flip = board_flip)

            # get all comments in the current board
            curr_comments = []
            for node in all_games[current_game].mainline():
                curr_comments.append(node.comment)
            
            if board_flip:
                for i in range(0, len(curr_comments) - 1, 2):
                    swap_comm = curr_comments[i]
                    curr_comments[i] = curr_comments[i+1]
                    curr_comments[i+1] = swap_comm

            # extract time information from the comments
            curr_times_p1, _, curr_remaining_times_p1, _ = extract_time_from_comment(curr_comments, timecontrol)

            # In the following, there are some sanity checks for the data and the final summary of the pgn import
            valid_data = True
            rem_time_counter = 0
            # check if remaining times are valid, if not, skip the game 
            for remaining_time in curr_remaining_times_p1:
                if remaining_time > timecontrol:
                    #print("Invalid data found - the present game will be skipped.")
                    rem_time_counter += 1
                    valid_data = False
            if rem_time_counter > 0: 
                valid_rem_time_counter += 1
                continue

            time_counter = 0
            # check if times_p1 are valid, if not, skip the game
            for time in curr_times_p1:
                if time < 0:
                    #print("Invalid data found - the present game will be skipped.")
                    time_counter += 1
                    valid_data = False
            if time_counter > 0: 
                valid_time_counter += 1
                continue

            outlier_counter = 0
            # check times for heavy outliers depending on the given timecontrol 
            for time in curr_times_p1:
                if time > (timecontrol * 0.15):
                    #print("Heavy outlier found - the present game will be skipped")
                    outlier_counter += 1
                    valid_data = False
            if outlier_counter > 0: 
                valid_outlier_counter +=1
                continue

            if valid_data:
                # Append all variables to final lists
                # Board Maps
                board_maps.extend(curr_board_maps)

                # Board Positions
                board_positions.extend(curr_board_positions_p1)

                # Materials W and B
                materials_p1.extend(curr_materials_p1)
                materials_p2.extend(curr_materials_p2)

                # Remaining Time W
                remaining_times_p1.extend(curr_remaining_times_p1)

                # Time W (prediction variable)
                times_p1.extend(np.array(curr_times_p1) / np.array(curr_remaining_times_p1))

                # Result
                curr_result = np.zeros_like(curr_times_p1)
                curr_result.fill(get_result_OHE(all_games[current_game].headers["Result"]))
                results.extend(curr_result)

                # print actual read games
                if((current_game % 10000 == 0) and (current_game != 0)): 
                    print("Games already read: ", current_game, "/", end)

    # get number of plies
    plies = len(times_p1)
    # flat board positions for creating final numpy ndarray
    for i, boardpos in enumerate(board_positions):
        board_positions[i] = np.array([boardpos.flatten()])
    
    # Lists to Numpy Arrays
    board_positions = np.array(board_positions).reshape(plies, 768)
    remaining_times_p1 = np.array(remaining_times_p1).reshape(plies,1)
    materials_p1 = np.array(materials_p1).reshape(plies,1)
    # materials_p2 = np.array(materials_p2).reshape(plies,1)
    times_p1 = np.array(times_p1).reshape(plies,1)
    results = np.array(results).reshape(plies, 1)


    # create return compressed data
    input_variables = np.hstack((materials_p1, board_positions)) # remaining_times_p1, materials_p2, results
    prediction_variable = times_p1
    board_maps = np.array(board_maps)
    
    print("-" * 30)
    print("Import Summmary")
    print("Games skipped due to invalid remaining times: ", valid_rem_time_counter)
    print("Games skipped due to invalid ply times: ", valid_time_counter)
    print("Games skipped due to outliers: ", outlier_counter)
    print("Number of plies which are imported: ", plies)

    return prediction_variable, input_variables, board_maps



""" Hyperparameters """
tc = 60
maxGames = 100000
year = "2020"
month = "01"

""" Directories """
pgn_dir = "/usr/src/app/datasets/lichess/" + year + "-" + month + "/Lichess_" + year + "-" + month + " (" + str(tc) + "+0).pgn"
zarr_dir = "datasets_zarr/" + year + "-" + month + "/Lichess_" + year + "-" + month + " (" + str(tc) + "+0)"

""" Function calls """
# get player variables
prediction_variable, input_variables, board_maps = read_pgn(pgn_dir, maxGames = maxGames, timecontrol = tc)
# save player variables
zarr.save(zarr_dir + "_predvar.zarr", prediction_variable)
zarr.save(zarr_dir + "_inputvar.zarr", input_variables)
zarr.save(zarr_dir + "_boardmaps.zarr", board_maps)

