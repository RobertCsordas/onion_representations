from framework.task import task, args, SimpleTask
from framework import dataset
import torch
import torch.nn.functional as F
from framework.interfaces import RecurrentResult
from typing import Dict, Tuple, Any, Optional
import framework
import torch.nn.utils.parametrizations


@args
def a(parser: framework.helpers.ArgumentParser):
    parser.add_argument("-state_size", default=128)
    parser.add_argument("-var_analysis.min_len", default=9)
    parser.add_argument("-var_analysis.max_len", default=9)
    parser.add_argument("-vecsum.gamma", default=0.4)


class Model(torch.nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, embedding_size: Optional[int] = None,
                 gamma: float = 0.4):
        super().__init__()

        embedding_size = embedding_size or hidden_size
        self.embedding = torch.nn.Embedding(vocab_size+1, embedding_size)
        self.sos_token = vocab_size
        self.hidden_size = hidden_size

        self.gamma = gamma
        self.fc = torch.nn.Linear(hidden_size, vocab_size)

    def encode(self, inp: torch.Tensor, in_len: torch.Tensor) -> torch.Tensor:
        x = self.embedding(inp.long())

        state = torch.zeros([inp.shape[1], self.hidden_size], device=x.device, dtype=x.dtype)
        gamma = 1

        for i in range(x.shape[0]):
            new_state = state + gamma * x[i]
            gamma = gamma * self.gamma

            state = torch.where((i < in_len)[..., None], new_state, state)

        return state

    def decode(self, encoded_state: torch.Tensor, outp: torch.Tensor, out_len: torch.Tensor) -> torch.Tensor:
        x = F.pad(outp[:-1], (0, 0, 1, 0), value=self.sos_token)
        x = self.embedding(x.long())

        state = encoded_state
        gamma = -1 / self.gamma

        states = []
        for i in range(x.shape[0]):
            new_state = state + gamma * x[i]
            gamma = gamma * self.gamma

            state = torch.where((i < out_len)[..., None], new_state, state)
            states.append(state)

        x = torch.stack(states, dim=0)

        return self.fc(x)



@task()
class VecsumRepeat(SimpleTask):
    def create_datasets(self):
        self.batch_dim = 1
        lens = (self.helper.args.var_analysis.min_len, self.helper.args.var_analysis.max_len)

        self.valid_sets.valid = dataset.RepeatCharDataset(30, 1, lens, 5000, seed=123, max_percent_per_lenght=0.20)
        self.train_set = dataset.RepeatCharDataset(30, 1, lens, 1000000, seed=0, exclude=[self.valid_sets.valid])

        self.valid_sets.valid_longer_1 = dataset.RepeatCharDataset(30, 1, (lens[1]+1, lens[1]+1), 1000, seed=674)
        self.valid_sets.valid_longer_2 = dataset.RepeatCharDataset(30, 1, (lens[1]+2, lens[1]+2), 1000, seed=345)
        if lens[1]+1 <= 30:
            self.valid_sets.valid_longer_1_norepeat = dataset.RepeatCharDataset(30, 1, (lens[1]+1, lens[1]+1), 1000, seed=345, with_repeats=False)

    def loss(self, logits: torch.Tensor, out: torch.Tensor, out_len: torch.Tensor) -> torch.Tensor:
        out = out.long()
        out = out.masked_fill(out_len[None] <= torch.arange(out.shape[0], device=out_len.device)[:, None], -1)
        return F.cross_entropy(logits.flatten(end_dim=-2), out.flatten(), ignore_index=-1)

    def run_model(self, data: Dict[str, torch.Tensor], ubatch: int = 0) -> RecurrentResult:
        encoded_state = self.model.encode(data["in"], data["in_len"])
        out = self.model.decode(encoded_state, data["out"], data["out_len"])

        loss = self.loss(out, data["out"], data["out_len"])
        return RecurrentResult(outputs=out, loss=loss), {}

    def run_model_validation(self, data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Any]:
        res, _ = self.run_model(data)
        return res.loss, (res.outputs.argmax(-1), data["out_len"])

    def create_model(self):
        return Model(
            len(self.train_set.in_vocabulary), self.helper.args.state_size,
            self.helper.args.var_analysis.embedding_size,
            gamma=self.helper.args.vecsum.gamma)

