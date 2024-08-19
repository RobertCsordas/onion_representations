from typing import Tuple, Optional
from matplotlib.pylab import RandomState
from framework.dataset import RepeatCharDataset


class RepeatCharFixedrepeatDataset(RepeatCharDataset):
    def generate_single(self, seed: RandomState):
        r = super().generate_single(seed).split()

        char_pos_to_repeat = seed.randint(len(r))
        char_to_repeat = r[char_pos_to_repeat]
        n_repeats = seed.randint(self.n_char_repeats[0], self.n_char_repeats[1]+1) - 1

        if n_repeats > 0:
            order = seed.permutation(len(r)-1)[:n_repeats]
            for i in order:
                if i == char_pos_to_repeat:
                    i = -1
                r[i] = char_to_repeat

        return " ".join(r)

    def __init__(self, n_char_repeats: Tuple[int,int], N_symbols: int, N_repeats: int, length: Tuple[int, int],
                 N: int, seed: int, cache_dir: str = "./cache/", max_percent_per_lenght: Optional[float] = None,
                 exclude=None):

        self.n_char_repeats = n_char_repeats

        if length[0] < self.n_char_repeats[-1]:
            raise ValueError("n_char_repeats must be smaller than the minimum length")

        super().__init__(N_symbols, N_repeats, length, N, seed, cache_dir, False, max_percent_per_lenght, exclude)