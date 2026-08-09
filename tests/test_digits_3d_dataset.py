import pickle
import runpy
from pathlib import Path

import numpy as np

MODULE = runpy.run_path(Path(__file__).parents[1] / "data/digits_3d/prepare.py")
VOCAB = MODULE["VOCAB"]
build_dataset = MODULE["build_dataset"]


def test_letters_are_in_vocab_but_absent_from_splits(tmp_path):
    build_dataset(tmp_path, train_repeats=3, val_repeats=2)

    with (tmp_path / "meta.pkl").open("rb") as handle:
        meta = pickle.load(handle)
    train = np.fromfile(tmp_path / "train.bin", dtype=np.uint16)
    val = np.fromfile(tmp_path / "val.bin", dtype=np.uint16)
    held_out_ids = {meta["stoi"][char] for char in "abcd"}

    assert "".join(meta["itos"][i] for i in range(meta["vocab_size"])) == VOCAB
    assert held_out_ids.isdisjoint(train)
    assert held_out_ids.isdisjoint(val)
    assert "".join(meta["itos"][int(i)] for i in train) == "0123456789" * 3
