from .gru_repeat import GruRepeat, RecurrentResult
from typing import Dict, Optional
import torch
from framework.task import task, args
from framework import dataset
import framework


@args
def a(parser: framework.helpers.ArgumentParser):
    parser.add_argument("-decode.n_layers", default=1)
    parser.add_argument("-decode.type", default="mlp", choice=["mlp", "godel", "gru"])
    parser.add_argument("-decode.gru.autoregressive", default=True)

class AutoregressiveDecoder:
    pass

class StateDecoderModel(torch.nn.Module):
    def __init__(self, hidden_size: int, n_symbols: int, max_len: int, n_layers: int):
        super().__init__()

        self.max_len = max_len

        self.model = torch.nn.Sequential()
        for i in range(n_layers-1):
            self.model.add_module(f"l{i}", torch.nn.Linear(hidden_size, hidden_size))
            self.model.add_module(f"r{i}", torch.nn.ReLU())

        self.model.add_module(f"l{n_layers-1}", torch.nn.Linear(hidden_size, n_symbols * max_len))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        res = self.model(h)
        return res.view(*res.shape[:-1], self.max_len, -1).transpose(0,1)


class GodelStateDecoderModel(AutoregressiveDecoder, torch.nn.Module):
    def __init__(self, hidden_size: int, n_symbols: int, max_len: int):
        super().__init__()

        self.max_len = max_len
        self.emb = torch.nn.Embedding(n_symbols, hidden_size)
        self.cls = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, n_symbols)
        )
        # self.cls = torch.nn.Linear(hidden_size, n_symbols)
        self.norm = torch.nn.LayerNorm(hidden_size)

        self.gamma = torch.nn.Parameter(torch.ones(hidden_size) * 0.9)
        self.g = torch.nn.Parameter(torch.ones(hidden_size))
        self.b = torch.nn.Parameter(torch.zeros(hidden_size))
        self.lin = torch.nn.Parameter(torch.ones(hidden_size) * 0.1)


    def forward(self, h: torch.Tensor, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = []
        for p in range(self.max_len if x is None else x.shape[0]):
            scale = self.g * (self.gamma.pow(p)) + p*self.lin + self.b

            l = self.cls(self.norm(h))
            logits.append(l)

            tok = l.argmax(dim=-1) if x is None else x[p].long()

            h = h - self.emb(tok) * scale

        return torch.stack(logits, dim=-2).transpose(0,1)


class GruDecoder(AutoregressiveDecoder, torch.nn.Module):
    def __init__(self, hidden_size: int, n_symbols: int, max_len: int, autoregressive: bool):
        super().__init__()

        self.max_len = max_len
        self.n_symbols = n_symbols
        if autoregressive:
            self.emb = torch.nn.Embedding(n_symbols+1, hidden_size)
        else:
            self.emb = None
        self.cls = torch.nn.Linear(hidden_size, n_symbols)

        self.gru = torch.nn.GRU(hidden_size, hidden_size, batch_first=True)


    def forward(self, h: torch.Tensor, tf: Optional[torch.Tensor]) -> torch.Tensor:
        logits = []

        if self.emb is not None:
            x = self.emb(torch.full((h.shape[0], 1), self.n_symbols, dtype=torch.long, device=h.device))
        else:
            x = torch.zeros(h.shape[0], 1, h.shape[-1], device=h.device)

        h = h.unsqueeze(0)

        for p in range(self.max_len if tf is None else tf.shape[0]):
            o, h = self.gru(x, h)
            l = self.cls(o)
            logits.append(l)
            if self.emb is not None:
                t = l.argmax(dim=-1) if tf is None else tf[p,...,None].long()
                x = self.emb(t)

        return torch.cat(logits, dim=-2).transpose(0,1)



@task()
class GruRepeatDecodeSeq(GruRepeat):
    def set_stage(self, stage: str):
        if self.stage == stage:
            return

        assert stage == "intervention"
        self.stage = stage
        self.create_optimizer()

        self.valid_sets = framework.data_structures.DotDict()
        lens = (self.helper.args.var_analysis.min_len, self.helper.args.var_analysis.max_len)
        self.valid_sets.valid = dataset.RepeatCharDataset(self.helper.args.var_analysis.n_symbols, 1, lens, 5000, seed=123, max_percent_per_lenght=0.20)
        self.create_loaders()

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)
        self.helper.saver.register("scaler", self.scaler, replace=True)

    def get_optimizer_param_list(self):
        if self.stage == "gru_training":
            return self.model.parameters()
        else:
            return self.decoder_model.parameters()

    def run_model_intervention(self, data: Dict[str, torch.Tensor], ubatch: int = 0) -> RecurrentResult:
        with torch.no_grad():
            encoded_state = self.model.encode(data["in"], data["in_len"])

        if isinstance(self.decoder_model, AutoregressiveDecoder):
            out = self.decoder_model(encoded_state[0], data["out"])
        else:
            out = self.decoder_model(encoded_state[0])

        out = out[:data["out"].shape[0]]

        loss = self.loss(out, data)
        return RecurrentResult(outputs=out, loss=loss), {}

    def create_model(self):
        if self.helper.args.decode.type == "mlp":
            self.add_model("decoder_model", StateDecoderModel(
                self.helper.args.state_size, len(self.train_set.in_vocabulary),self.helper.args.var_analysis.max_len,
                self.helper.args.decode.n_layers))
        elif self.helper.args.decode.type == "godel":
            self.add_model("decoder_model", GodelStateDecoderModel(
                self.helper.args.state_size, len(self.train_set.in_vocabulary),self.helper.args.var_analysis.max_len))
        elif self.helper.args.decode.type == "gru":
            self.add_model("decoder_model", GruDecoder(
                self.helper.args.state_size, len(self.train_set.in_vocabulary),self.helper.args.var_analysis.max_len,
                autoregressive=self.helper.args.decode.gru.autoregressive))
        else:
            raise ValueError(f"Unknown decoder type {self.helper.args.decode.type}")

        self.stage = "gru_training"

        return self.create_rnn()