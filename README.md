# Time Management in Chess with Neural Networks and Human Data


<p align="center">  
  Student research project in cooperation with TU Darmstadt. <br> 
  Written project submission with a detailed explanation of the methods used can be found on the homepage of <a href="https://ml-research.github.io/">AIML</a> soon. 
</p>

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/de/thumb/2/24/TU_Darmstadt_Logo.svg/1200px-TU_Darmstadt_Logo.svg.png" width="200" title="Logo TU Darmstadt">
</p>

## Abstract

Time management in two-player board games such as chess describes the efficient use of the available playing time to generate the best possible moves without losing the game due to lack of time.
The present work addresses the limited application of machine learning and deep learning methods for this research topic. Previous time managers are based on heuristics and rarely use more complex methods. It is investigated how computer chess engines can benefit from the human gaming behaviour in terms of the allocation of time for each move. For this purpose, three neural networks are presented that learn the human behaviour of time allocation and reproduce it in chess games. One of these neural networks emerges as the most suitable for this problem.  A generic implementation of this neural network into the game process of two chess engines is developed which can be applied to any UCI-ready chess engine and to any time control. The neural network thus acts as a time manager without applying well-known heuristics and without being built into the specific search process of a chess engine.
The developed neural networks which are designed to handle the art of time management are called Chess Time Management Network (CTM Net) 1.0, 2.0 and 2.1. Although no significant and positive ELO difference for CTM Net was found, the present work is the first of its kind and represents an important groundwork in the development of modern time managers. CTM Net is tested with state-of-the-art chess engines such as Stockfish to generate comparability and classification of the results in today's process of chess AI.

### Dependencies

The present work is programmed in Python 3.8.8. 

* All required packages are listed in requirements.txt. 
* For the plots, [SciencePlots](https://github.com/garrettj403/SciencePlots) is used in the submission. This package is left out for the GitHub publication.
* Further, [CuteChess-cli](https://github.com/cutechess/cutechess) is needed to be able to start the gaming process of two engines. 
* For extractions of databases, [pgn-extract](https://www.cs.kent.ac.uk/people/staff/djb/pgn-extract/) is used. 


### Start the gaming process of two engines

* To start the game process of two chess engines, a python script (```CutechessCommunication```) was created. In this script, one can choose the engines and further options for configurating the game process. If one engine is choosen, the engine should be used in ```parserCTMNet_2.py``` too. 
* As only CTM Net 2.1 was used for the submission, the parser file is optimized for CTM Net 2.1. For the other models, one has to modify the file regarding the scalers used and the shapes of the input features.
* As the (re-)distribution of the chess engines used is not allowed, only the empty folder ```engines``` (inside the folder ```chess```) is included in which any UCI-ready chess engine can be placed. The same holds for [Syzygy tablebases](https://syzygy-tables.info/)  and the [Anti-Draw Openings](https://www.sp-cc.de/anti-draw-openings.htm) used. 


### Training a model

* The training data is located in the folder ```datasets_zarr```. As the training data size is too big for Github, the training data has to be created first by using ```converter_pgn_zarr.py```. The ```dataset_urls.txt``` and ```filtertags.txt``` are included in this repository to be able to download and extract the datasets which are used in the submission. The data converted to zarr-arrays will be used as input by ```main.py``` and ```main_v2.py```.
  1. ```cd datasets/lichess/```
  2. ```wget -i dataset_urls.txt```
  3. The downloaded files should have the name "lichess_db_standard_rated_year-month.pgn.bz2" and should be moved to folders named "year-month" respectively (these folders are not created yet). As the files are still compressed (bz2), decompress them (e.g. by ```bzip2 -d filename.bz2```)
  4. Open ```extractDatabases.py```and adjust the parameters if needed. 
  5. Start the extraction process via ```python3 extractDatabases.py``` and manually move the files to the direction you want them to be. It is recommended to rename every extracted .pgn-file as "Lichess_year-month (60+0).pgn".
  6. Open ```converter_pgn-zarr.py``` and adjust the hyperparameters, directories and function calls at the bottom of the file if needed. 
  7. Start the converting process via ```python3 converter_pgn-zarr.py```. This python file creates zarr arrays out of the pre-filtered .pgn-files in the previous step. Further, the data is checked for validity by this file. 
  8. The extracted and converted files should be located in the folder ```datasets_zarr``` (if not, move them manually). The folder layout should be as follows:
     ```
     datasets_zarr
        ---2020-01
            --- Lichess_2020-01 (60+0)_boardmaps.zarr
            --- Lichess_2020-01 (60+0)_inputvar.zarr
            --- Lichess_2020-01 (60+0)_predvar.zarr
        ---2020-02
            --- ...
     ```
* To train the model(s), ```main.py``` is used for CTM Net 1.0. For CTM Net 2.0 and 2.1, ```main_v2.py``` is used due to the different shape of the input features.
* Scalers for the normalization of the input and output features are located in the folder ```scalers``` and are generated automatically by ```main.py``` and ```main_v2.py```. If a trained model should be used in the gaming process, the scalers has to be moved manually to the corresponding folder in the folder ```chess``` which is also named ```scalers```. The same applies for the PyTorch-models. 
* Start the training via:
  ```
  python3 main_v2.py
  ```


## Authors
Tillmann Rheude

Supervisor at TU Darmstadt: Johannes Czech


## Version History

* 1.0
    * Submission of the research project

## License

MIT License

## Acknowledgments

* [Lichess](https://lichess.org/)
* [pgn-extract](https://www.cs.kent.ac.uk/people/staff/djb/pgn-extract/)
* [python-chess](https://python-chess.readthedocs.io/en/latest/)
