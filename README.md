# patzer-ccc26

A bitboard chess board compiled with Numba, for the perft framework of the Cogsci Chess Club (Winter 2026) at Osnabrück University.

It needs Python 3.12 or newer, an x86-64 CPU with BMI2 (Intel Haswell or AMD Excavator and later, slow on AMD before Zen 3, whose PEXT is microcoded), and the packages in `requirements.txt` next to the course's own:

```sh
pip install -r src/ccc26/user/dobrockmann/requirements.txt
CHESS_BOARD_CLASS=ccc26.user.dobrockmann.adapter.PatzerBoard python -m unittest tests/test_perft.py
```

[perft.txt](perft.txt) holds the course's perft against python-chess on the positions of its tests, at depth 4, 3 runs each, with the machine it ran on. Each comes from one command, here for Kiwi-Pete:

```sh
python -m ccc26.eval.perft --board ccc26.user.dobrockmann.adapter.PatzerBoard --depth 4 --fen "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
```

[patzer-perft.txt](patzer-perft.txt) holds our own perft on the same positions, at depth 4, 3 runs each, with the same counts as python-chess. Its header states how it counts, which differs from the course's loop, so the times don't compare. For Kiwi-Pete:

```sh
python -m ccc26.user.dobrockmann.board.cli --depth 4 --fen "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
```

Two profiles show where the time goes when the course's perft runs our board alone on Kiwi-Pete. Both profilers slow every Python call, not the compiled code, so their times run longer than in perft.txt.

[pyinstrument.txt](pyinstrument.txt) is a call tree from [pyinstrument](https://github.com/joerick/pyinstrument) (pyinstrument==5.1.3), which samples and so slows the run less:

```sh
python -m pyinstrument -o pyinstrument.txt -m ccc26.eval.perft --board ccc26.user.dobrockmann.adapter.PatzerBoard --mode custom --depth 4 --fen "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
```

[cprofile.txt](cprofile.txt) holds the first 40 rows of Python's built-in cProfile, which counts every call:

```sh
python -m cProfile -s cumulative -m ccc26.eval.perft --board ccc26.user.dobrockmann.adapter.PatzerBoard --mode custom --depth 4 --fen "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1" > cprofile.txt
```
