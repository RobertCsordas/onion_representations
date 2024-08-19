from typing import Dict, Any, Tuple, List, Optional
import string
import numpy as np
from .helpers import CacheLoaderMixin
from .sequence import TextSequenceTestState
from tqdm import tqdm
import torch.utils.data
import math
import hashlib
from ..utils import rejection_sample
from .. import data_structures


class RepeatCharDataset(CacheLoaderMixin, torch.utils.data.Dataset):
    in_vocabulary = None

    def config_id(self) -> str:
        return f"{self.N_symbols}_{self.N_repeats}_{self.N}_{self.len[0]}_{self.len[1]}_{self.with_repeats}_{self.max_percent_per_lenght}_{self.dep_postfix}_{self.seed}"

    def generate_single(self, seed: np.random.RandomState):
        if self.len[0] == self.len[1]:
            len = self.len[0]
        else:
            len = seed.randint(max(self.len[0], 2), self.len[1]+1)

        if self.with_repeats:
            rnd = seed.randint(self.N_symbols, size=(len,))
        else:
            rnd = seed.permutation(self.N_symbols)[:len]

        return " ".join(self.symbols[i] for i in rnd.tolist())

    def generate_dataset(self) -> Tuple[List[int], List[int]]:
        in_sentences = []
        out_sentences = []

        print(f"Generating {self.__class__.__name__} with {self.N} elements and {self.N_symbols} symbols")

        if self.exclude is not None:
            print("Calculating exclude list. It might take a while...")
            exclude = set()
            for e in self.exclude:
                for i in tqdm(e.in_sequences):
                    exclude.add(hash(e.in_vocabulary.to_string(i)))
            print(f"Excluded {len(exclude)} elements.")
        else:
            exclude = set()

        samples = set()
        if self.len[0] == 1:
            for i in range(self.N_symbols):
                samples.add(self.symbols[i])

        if self.max_percent_per_lenght is not None:
            # Limit the possbile number of samples per length to a certain percentage of possibilities.
            n_remaining = {}
            for i in range(self.len[0], self.len[1] + 1):
                if self.with_repeats:
                    n_remaining[i] = int(self.N_symbols ** i * self.max_percent_per_lenght)
                else:
                    n_remaining[i] = int((math.factorial(self.N_symbols) / math.factorial(self.N_symbols - i)) * self.max_percent_per_lenght)

            def test(x):
                lx = len(x.split(" "))
                if n_remaining[lx] > 0:
                    n_remaining[lx] -= 1
                    return True
                else:
                    return False
        else:
            test = lambda x: True

        samples = samples.union(rejection_sample(self.N - (self.N_symbols if self.len[0] == 1 else 0),
                                np.random.RandomState(self.seed), self.generate_single, test=test, exclude=exclude))

        print("Postprocessing...")

        myhash = hashlib.md5()

        for s in tqdm(samples):
            myhash.update(s.encode("utf-8"))
            s = self.in_vocabulary(s)
            in_sentences.append(s)
            out_sentences.append(sum([[x] * self.N_repeats for x in s], []))

        myhash = myhash.hexdigest()
        return in_sentences, out_sentences, myhash

    def __getitem__(self, item: int) -> Dict[str, Any]:
        return {
            "in": np.asarray(self.in_sequences[item], np.uint8),
            "out": np.asarray(self.out_sequences[item], np.uint8),
            "in_len":  np.asarray(len(self.in_sequences[item]), dtype=np.int32),
            "out_len": np.asarray(len(self.in_sequences[item]), dtype=np.int32),
        }

    def init_vocabulary(self):
        if self.in_vocabulary is None:
            self.__class__.in_vocabulary = data_structures.WordVocabulary(list(self.symbols))
            self.__class__.out_vocabulary = self.in_vocabulary

    def __init__(self, N_symbols: int, N_repeats: int, length: Tuple[int, int], N: int, seed: int,
                 cache_dir: str = "./cache/", with_repeats: bool = True,
                 max_percent_per_lenght: Optional[float] = None, exclude=None):
        self.cache_dir = cache_dir
        self.N = N
        self.N_symbols = N_symbols
        self.N_repeats = N_repeats
        self.len = length
        self.seed = seed
        self.with_repeats = with_repeats
        self.symbols = (string.digits + string.ascii_letters)[:N_symbols]
        self.max_percent_per_lenght = max_percent_per_lenght
        self.exclude=exclude

        if self.exclude is not None:
            hashes = list(sorted(e.hash for e in self.exclude))
            self.dep_postfix = "_e".join(hashes)
        else:
            self.dep_postfix = ""

        if (not self.with_repeats) and len(self.symbols) < length[1]:
            raise ValueError("Not enough symbols.")

        self.init_vocabulary()

        self.in_sequences, self.out_sequences, self.hash = self.load_cache()

    def get_output_size(self) -> int:
        return len(self.out_vocabulary)

    def get_input_size(self) -> int:
        return len(self.in_vocabulary)

    @property
    def max_in_len(self) -> int:
        return self.len[1]

    @property
    def max_out_len(self) -> int:
        return self.len[1]

    def __len__(self) -> int:
        return len(self.in_sequences)

    def start_test(self) -> TextSequenceTestState:
        return TextSequenceTestState(lambda x: " ".join(self.in_vocabulary(x)),
                                     lambda x: " ".join(self.out_vocabulary(x)))


class RepeatCharDirectDataset(RepeatCharDataset):
    def init_vocabulary(self):
        if self.in_vocabulary is None:
            syms = list(self.symbols) + ["SEP", "-"]
            self.__class__.in_vocabulary = data_structures.WordVocabulary(syms, split_punctuation=False)
            self.__class__.out_vocabulary = self.in_vocabulary

    def __getitem__(self, item: int) -> Dict[str, Any]:
        res = super().__getitem__(item)
        inp = np.empty([res["in_len"] + res["out_len"] + 1], np.uint8)
        inp[:res["in_len"]] = res["in"]
        inp[res["in_len"]] = self.in_vocabulary["SEP"]
        inp[res["in_len"] + 1:] = self.in_vocabulary["-"]

        outp = np.empty_like(inp)
        outp[:res["in_len"]] = self.out_vocabulary["-"]
        outp[res["in_len"]] = self.out_vocabulary["SEP"]
        outp[res["in_len"] + 1:] = res["out"]

        l = res["in_len"] + res["out_len"] + 1
        assert res["in_len"] == res["out_len"]

        return {
            "in": inp,
            "out": outp,
            "in_len": np.asarray(l, dtype=np.int32),
            "out_len": np.asarray(l, dtype=np.int32)
        }

    def start_test(self) -> TextSequenceTestState:
        return TextSequenceTestState(lambda x: " ".join(self.in_vocabulary(x)),
                                     lambda x: " ".join(self.out_vocabulary(x)),
                                     add_eos=False)


class RepeatCharAutoregressiveDataset(RepeatCharDataset):
    def init_vocabulary(self):
        if self.in_vocabulary is None:
            syms = list(self.symbols) + ["SEP"]
            self.__class__.in_vocabulary = data_structures.WordVocabulary(syms, split_punctuation=False)
            self.__class__.out_vocabulary = self.in_vocabulary

    def __init__(self, N_symbols: int, N_repeats: int, length: Tuple[int, int], N: int, seed: int,
                 cache_dir: str = "./cache/", ignore_index: int = -100):
        super().__init__(N_symbols, N_repeats, length, N, seed, cache_dir)
        self.ignore_index = ignore_index

    def __getitem__(self, item: int) -> Dict[str, Any]:
        res = super().__getitem__(item)

        l = res["in_len"] + res["out_len"] + 1

        inp = np.empty([l], np.int16)
        inp[:res["in_len"]] = res["in"]
        inp[res["in_len"]] = self.in_vocabulary["SEP"]
        inp[res["in_len"] + 1:] = res["out"]

        outp = np.empty_like(inp)
        outp[:res["in_len"]] = self.ignore_index
        outp[res["in_len"]:-1] = res["out"]
        outp[-1] = self.out_vocabulary["SEP"]

        return {
            "in": inp,
            "out": outp,
            "in_len": np.asarray(l, dtype=np.int32),
            "out_len": np.asarray(l, dtype=np.int32)
        }

    def start_test(self) -> TextSequenceTestState:
        return TextSequenceTestState(lambda x: " ".join(self.in_vocabulary(x)),
                                     lambda x: " ".join(self.out_vocabulary(x)),
                                     add_eos=False, ignore_index=self.ignore_index)

