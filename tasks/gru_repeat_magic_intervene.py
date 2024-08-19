from .gru_repeat import GruRepeat, RecurrentResult
from typing import Dict
import torch
from framework.task import task, args
from framework import dataset
import framework
import torch.nn.functional as F
from datasets import RepeatCharFixedrepeatDataset

@args
def a(parser: framework.helpers.ArgumentParser):
    parser.add_argument("-decode.n_layers", default=1)
    parser.add_argument("-magic_intervention.disable_scale", default=False)
    parser.add_argument("-magic_intervention.lin", default=True)


class MagicIntervention(torch.nn.Module):
    def __init__(self, hidden_size: int, n_symbols: int, n_max_pos: int, disable_scale: bool = False,
                 lin: bool = True):
        super().__init__()

        self.embedding = torch.nn.Embedding(n_symbols, hidden_size)

        if disable_scale:
            self.register_buffer("gamma", torch.ones(hidden_size))
        else:
            self.gamma = torch.nn.Parameter(torch.ones(hidden_size) * 0.9)
            #self.gamma = torch.nn.Parameter(torch.ones(1) * 0.5)

        if lin and not disable_scale:
            self.lin = torch.nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_buffer("lin", torch.zeros(hidden_size))

        self.g = torch.nn.Parameter(torch.ones(hidden_size))
        self.b = torch.nn.Parameter(torch.zeros(hidden_size))

    def forward(self, h: torch.Tensor, old_token: torch.Tensor, new_token: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        scale = self.g * (self.gamma.pow(pos[:,None])) + pos[:, None]*self.lin + self.b

        x = (self.embedding(old_token.long())).tanh()
        y = (self.embedding(new_token.long())).tanh()

        return h + (y - x) * scale


@task()
class GruRepeatMagicIntervention(GruRepeat):
    def set_stage(self, stage: str):
        if self.stage == stage:
            return

        assert stage == "intervention"
        self.stage = stage
        self.create_optimizer()

        self.valid_sets = framework.data_structures.DotDict()
        lens = (self.helper.args.var_analysis.min_len, self.helper.args.var_analysis.max_len)
        self.valid_sets.intervention_valid = dataset.RepeatCharDataset(self.helper.args.var_analysis.n_symbols, 1, lens, 5000, seed=123, max_percent_per_lenght=0.20)

        for nrep in [3]:
            lens = (max(self.helper.args.var_analysis.min_len, nrep), self.helper.args.var_analysis.max_len)
            self.valid_sets[f"valid_rep_{nrep}"] = RepeatCharFixedrepeatDataset((nrep, nrep), self.helper.args.var_analysis.n_symbols, 1,
                                                                                lens, 1000, seed=123+nrep)
            self.valid_sets[f"valid_norep_{nrep}"] = dataset.RepeatCharDataset(self.helper.args.var_analysis.n_symbols, 1, lens, 1000, seed=123+nrep, max_percent_per_lenght=0.20)

        self.create_loaders()

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)
        self.helper.saver.register("scaler", self.scaler, replace=True)


    def run_model_intervention(self, data: Dict[str, torch.Tensor], ubatch: int = 0) -> RecurrentResult:
        replace_pos = (torch.rand(data["in"].shape[1], device=data["in"].device) * data["in_len"]).type(torch.long)
        old_token = data["in"][replace_pos, torch.arange(data["in"].shape[1])]

        new_token = torch.randint_like(old_token, 0, len(self.train_set.in_vocabulary))

        modified_in = data["in"].clone()
        modified_in[replace_pos, torch.arange(data["in"].shape[1])] = new_token

        with torch.no_grad():
            encoded_state = self.model.encode(modified_in, data["in_len"])

        modified_state = self.intervention_model(encoded_state, new_token, old_token, replace_pos)

        out = self.model.decode(modified_state, data["out"], data["out_len"])

        loss = self.loss(out, data)
        return RecurrentResult(outputs=out, loss=loss), {}

    def create_model(self):
        self.add_model("intervention_model", MagicIntervention(
            self.helper.args.state_size, len(self.train_set.in_vocabulary),
            self.helper.args.var_analysis.max_len,
            disable_scale=self.helper.args.magic_intervention.disable_scale))

        self.stage = "gru_training"

        return self.create_rnn()