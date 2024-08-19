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
    parser.add_argument("-var_analysis.embedding_size", default="none", parser=parser.int_or_none_parser)
    parser.add_argument("-var_analysis.var_size", default="none", parser=parser.int_or_none_parser)
    parser.add_argument("-var_analysis.model_train_steps", default=30000)
    parser.add_argument("-var_analysis.min_len", default=9)
    parser.add_argument("-var_analysis.max_len", default=9)
    parser.add_argument("-var_analysis.n_symbols", default=30)
    parser.add_argument("-var_analysis.mask", default=False)
    parser.add_argument("-var_analysis.no_input", default=False)
    parser.add_argument("-var_analysis.ngram", default=1)
    parser.add_argument("-var_analysis.mask_reg", default=0.001)
    parser.add_argument("-var_analysis.intervention_model", default="autosegment", choice=["fixed", "autosegment"])


class Model(torch.nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, embedding_size: Optional[int] = None,
                 no_input: bool = False):
        super().__init__()

        embedding_size = embedding_size or hidden_size
        self.no_input = no_input
        self.embedding = torch.nn.Embedding(vocab_size+1+int(self.no_input), embedding_size)
        self.sos_token = vocab_size
        self.no_input_token = self.sos_token + 1

        self.rnn = torch.nn.GRU(embedding_size, hidden_size)
        self.fc = torch.nn.Linear(hidden_size, vocab_size)

    def encode(self, inp: torch.Tensor, in_len: torch.Tensor) -> torch.Tensor:
        in_len, idx = in_len.sort(dim=0, descending=True)
        _, reverse_idx = idx.sort(dim=0, descending=False)

        x = inp[:, idx]
        x = self.embedding(x.long())

        x = torch.nn.utils.rnn.pack_padded_sequence(x, in_len.cpu())
        x, encoded_state = self.rnn(x)

        return encoded_state[:, reverse_idx]

    def decode(self, encoded_state: torch.Tensor, outp: torch.Tensor, out_len: torch.Tensor) -> torch.Tensor:
        if self.no_input:
            outp = torch.full_like(outp, self.no_input_token)

        out_len, idx = out_len.sort(dim=0, descending=True)
        _, reverse_idx = idx.sort(dim=0, descending=False)
        encoded_state = encoded_state[:, idx]

        x = outp[:, idx]
        x = F.pad(x[:-1], (0, 0, 1, 0), value=self.sos_token)
        x = self.embedding(x.long())

        x = torch.nn.utils.rnn.pack_padded_sequence(x, out_len.cpu())
        x, _ = self.rnn(x, encoded_state)

        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x)

        x = x[:, reverse_idx]
        return self.fc(x)


class Intervention:
    def __init__(self, max_len: int):
        self.n_variables = max_len

    def corrupt(self, data: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor]]:
        raise NotImplementedError()

    def fix(self, variables: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def get_reordering(self, batch_size: int, device: torch.device) -> Optional[torch.Tensor]:
        return None

    def get_n_variables(self) -> int:
        return self.n_variables


class ElementwiseIntervention(Intervention):
    def corrupt(self, data: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor]]:
        batch_size = data["in"].shape[1]

        if batch_size % 2 != 0:
            raise ValueError("n must be even")

         # Create a symmetric random pairing
        p = torch.randperm(batch_size, device=data["in"].device)
        i1 = p[:batch_size//2]
        i2 = p[batch_size//2:]
        pairing = torch.empty_like(p)
        pairing[i1] = i2
        pairing[i2] = i1

        # Create the intervention mask in the "virtual order" defined by p
        intervention_limit = torch.minimum(data["in_len"][i1], data["in_len"][i2])

        # intervene_on = (torch.rand(batch_size // 2, device=data["in"].device) * intervention_limit.float()).long()
        # intervention_mask_raw = torch.zeros([data["in"].shape[0], batch_size // 2], device=data["in"].device, dtype=torch.bool)
        # intervention_mask_raw.scatter_(0, intervene_on[None], True)

        # Intervene on random elements
        intervention_mask_raw = torch.rand([data["in"].shape[0], batch_size // 2], device=data["in"].device) < 0.5
        intervention_mask_raw.masked_fill_(torch.arange(data["in"].shape[0], device=data["in"].device)[:, None] >= intervention_limit[None], False)

        # Create the real, symmetric mask
        intervention_mask = torch.zeros_like(data["in"], dtype=torch.bool)
        intervention_mask[:, i1] = intervention_mask_raw
        intervention_mask[:, i2] = intervention_mask_raw

        # Modify the input. Then intervene on the state to restore the original output. This way it is compatible
        # with the standard validation functions
        result = dict(data)
        result["in"] = torch.where(intervention_mask, data["in"][:, pairing], data["in"])

        # Save state
        self.pairing = torch.argsort(pairing)
        self.intervention_mask = intervention_mask

        return result

    def fix(self, variables: torch.Tensor) -> torch.Tensor:
        res = torch.where(self.intervention_mask.t()[..., None], variables[self.pairing], variables)
        self.pairing = None
        self.intervention_mask = None
        return res


class NgramIntervention(Intervention):
    def __init__(self, n: int, n_symbols: int, max_len: int):
        self.n_symbols = n_symbols
        self.n = n
        self.n_variables = max_len - n + 1

    def corrupt(self, data: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor]]:
        batch_size = data["in"].shape[1]

        # Intervene on random elements
        length_mask = torch.arange(data["in"].shape[0], device=data["in"].device)[:, None] >= data["in_len"][None]

        intervene_on = (torch.rand(batch_size, device=data["in"].device) * data["in_len"].float()).long()
        intervention_mask = torch.zeros_like(data["in"], dtype=torch.bool)
        intervention_mask.scatter_(0, intervene_on[None], True)

        # Create masks for moving variables and inputs
        convin = intervention_mask.T.unsqueeze(-2).float()
        input_move_mask = F.conv1d(convin, torch.ones(2*self.n-1, device=convin.device, dtype=torch.float)[None,None], padding=self.n-1).squeeze(-2).T > 0
        var_move_mask = F.conv1d(convin, torch.ones(2*self.n-2, device=convin.device, dtype=torch.float)[None,None], padding=2*self.n-3).squeeze(-2)[:, 2*self.n-3:].T > 0
        assert var_move_mask.shape[0] >= self.n_variables
        var_move_mask = var_move_mask[:self.n_variables]

        # If you intervene on a variable, and pairs are stored, the next one should be also intervened on
        input_move_mask.masked_fill_(length_mask, False)
        # var_move_mask.masked_fill_(length_mask[self.n-1:], False)

        # Modify the input. Then intervene on the state to restore the original output. This way it is compatible
        # with the standard validation functions
        result = dict(data)

        randinput = torch.randint_like(data["in"], 0, self.n_symbols)

        corrupted_input = torch.where(intervention_mask, randinput, data["in"])
        var_source = torch.where(input_move_mask, data["in"], randinput)

        result["in"] = torch.cat([corrupted_input, var_source], dim=1)
        result["in_len"] = result["in_len"].repeat([2])

        # Save state
        self.var_move_mask = var_move_mask

        return result

    def fix(self, variables: torch.Tensor) -> Tuple[torch.Tensor]:
        to_fix, var_src = variables.chunk(2, 0)
        res = torch.where(self.var_move_mask.T[..., None], var_src, to_fix)
        self.var_move_mask = None
        return res

    def get_reordering(self, batch_size: int, device: torch.device) -> Optional[torch.Tensor]:
        return torch.arange(0, batch_size, device=device)

    def get_n_variables(self) -> int:
        return self.n_variables


class InterventionModel(framework.layers.RegularizedLayer, framework.layers.LayerWithVisualization, torch.nn.Module):
    def __init__(self, hidden_size: int, var_size: int, n_variables: int, masking: bool, reg_loss: float):
        super().__init__()

        self.var_size = var_size
        self.masking = masking
        self.intervention_size = (hidden_size // var_size) * var_size
        self.reg_loss = reg_loss

        self.proj = torch.nn.utils.parametrizations.orthogonal(
            torch.nn.Linear(hidden_size, hidden_size, bias=False)
        )

        if self.masking:
            self.mask = torch.nn.Parameter(torch.zeros(self.intervention_size))

    def to_semnatic_space(self, x: torch.Tensor) -> torch.Tensor:
        self.proj_x = self.proj(x)
        return self.proj_x[..., :self.intervention_size].view(*self.proj_x.shape[:-1], -1, self.var_size)

    def from_semnatic_space(self, x: torch.Tensor, reordering: Optional[torch.Tensor]) -> torch.Tensor:
        x = x.flatten(start_dim=-2)

        proj_x = self.proj_x[reordering] if reordering is not None else self.proj_x
        self.proj_x = None

        if self.masking:
            x = self.combine(x, proj_x[..., :self.intervention_size])

        x = torch.cat([x, proj_x[..., self.intervention_size:]], dim=-1)
        return F.linear(x, self.proj.weight.t())

    def combine(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.training:
            mask = framework.layers.gumbel_sigmoid(self.mask, hard=True)
        else:
            mask = (self.mask > 0).float()

        self.add_reg(lambda: self.reg_loss * self.mask.sum(), "mask_reg")
        return mask * x + (1 - mask) * y

    def plot(self, options: Dict[str, Any]) -> Dict[str, Any]:
        vars = (self.mask > 0).view(-1, self.var_size).T.float()
        return {
            "mask": framework.visualize.plot.Heatmap(vars, xlabel="Variable", ylabel="Channel", textval=False,
                                                     range=(0,1))
        }


class AutosegmentationInterventionModel(framework.layers.RegularizedLayer, framework.layers.LayerWithVisualization, torch.nn.Module):
    def __init__(self, hidden_size: int, var_size: int, n_variables: int, reg_loss: float):
        torch.nn.Module.__init__(self)
        framework.layers.LayerWithVisualization.__init__(self)
        framework.layers.RegularizedLayer.__init__(self)

        self.n_variables = n_variables
        self.var_size = var_size
        self.intervention_size = (hidden_size // var_size) * var_size
        self.reg_loss = reg_loss

        self.proj = torch.nn.utils.parametrizations.orthogonal(
            torch.nn.Linear(hidden_size, hidden_size, bias=False)
        )

        self.mask = torch.nn.Parameter(torch.randn(hidden_size, n_variables+1)*0.01)

    def to_semnatic_space(self, x: torch.Tensor) -> torch.Tensor:
        self.proj_x = self.proj(x)
        return self.proj_x.unsqueeze(-2).expand([*([-1]*(x.ndim - 1)), self.n_variables, -1])

    def get_deterministic_mask(self):
        return F.one_hot(self.mask.argmax(-1), self.mask.shape[1]).float()

    def from_semnatic_space(self, x: torch.Tensor, reordering: Optional[torch.Tensor]) -> torch.Tensor:
        proj_x = self.proj_x[reordering] if reordering is not None else self.proj_x
        self.proj_x = None

        x = torch.cat([
            x,
            proj_x.unsqueeze(-2)
        ], dim=-2)

        if self.training:
            masks = framework.layers.gumbel_softmax(self.mask, hard=True)
            self.add_reg(lambda: self.reg_loss * (self.mask[:, :-1].mean() - self.mask[:, -1].mean()), "mask_reg")
        else:
            masks = self.get_deterministic_mask()

        x = torch.einsum("bvc,cv->bc", x, masks)
        return F.linear(x, self.proj.weight.t())


    def plot(self, options: Dict[str, Any]) -> Dict[str, Any]:
        mask = self.get_deterministic_mask()
        return {
            "mask": framework.visualize.plot.Heatmap(mask, xlabel="Variable", ylabel="Channel", textval=False,
                                                     range=(0,1)),
            "var_sizes": framework.visualize.plot.Barplot(mask.sum(0), xlabel="Variable", ylabel="Size")
        }



@task()
class GruRepeat(SimpleTask):
    def create_state(self):
        self.change_stage(True)

    def create_datasets(self):
        self.batch_dim = 1
        lens = (self.helper.args.var_analysis.min_len, self.helper.args.var_analysis.max_len)

        self.valid_sets.valid = dataset.RepeatCharDataset(self.helper.args.var_analysis.n_symbols, 1, lens, 5000, seed=123, max_percent_per_lenght=0.20)
        self.train_set = dataset.RepeatCharDataset(self.helper.args.var_analysis.n_symbols, 1, lens, 1000000, seed=0, exclude=[self.valid_sets.valid])

        self.valid_sets.valid_longer_1 = dataset.RepeatCharDataset(self.helper.args.var_analysis.n_symbols, 1, (lens[1]+1, lens[1]+1), 1000, seed=674)
        self.valid_sets.valid_longer_2 = dataset.RepeatCharDataset(self.helper.args.var_analysis.n_symbols, 1, (lens[1]+2, lens[1]+2), 1000, seed=345)
        if lens[1]+1 <= self.helper.args.var_analysis.n_symbols:
            self.valid_sets.valid_longer_1_norepeat = dataset.RepeatCharDataset(self.helper.args.var_analysis.n_symbols, 1, (lens[1]+1, lens[1]+1), 1000, seed=345, with_repeats=False)

    def loss(self, logits: torch.Tensor, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = data["out"].long()
        out_len = data["out_len"]

        if "mask" in data:
            out = out.masked_fill(~data["mask"].bool(), -1)

        out = out.masked_fill(out_len[None] <= torch.arange(out.shape[0], device=out_len.device)[:, None], -1)
        return F.cross_entropy(logits.flatten(end_dim=-2), out.flatten(), ignore_index=-1)

    def run_gru_model(self, data: Dict[str, torch.Tensor], ubatch: int = 0) -> RecurrentResult:
        encoded_state = self.model.encode(data["in"], data["in_len"])
        out = self.model.decode(encoded_state, data["out"], data["out_len"])

        loss = self.loss(out, data)
        return RecurrentResult(outputs=out, loss=loss), {}

    def run_model_validation(self, data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Any]:
        res, _ = self.run_model(data)
        return res.loss, (res.outputs.argmax(-1), data["out_len"])

    def run_model_intervention(self, data: Dict[str, torch.Tensor], ubatch: int = 0) -> RecurrentResult:

        # Corrupt data
        data_intervened = self.intervention.corrupt(data)

        # Encode
        encoded_state = self.model.encode(data_intervened["in"], data_intervened["in_len"])
        encoded_state = encoded_state.permute([1,0,2]).flatten(start_dim=-2)

        # Project to the semantically meaningful space and do the intervention
        vars = self.intervention_model.to_semnatic_space(encoded_state)

        # Fix the corrupted data
        vars = self.intervention.fix(vars)
        reordering = self.intervention.get_reordering(data["in"].shape[1], data["in"].device)

        # Project back
        encoded_state_intervention = self.intervention_model.from_semnatic_space(vars, reordering)

        # Decode
        encoded_state_intervention = encoded_state_intervention.view(encoded_state_intervention.shape[0], -1, self.helper.args.state_size)
        encoded_state_intervention = encoded_state_intervention.permute([1,0,2])

        out = self.model.decode(encoded_state_intervention, data["out"], data["out_len"])
        loss = self.loss(out, data)

        return RecurrentResult(outputs=out, loss=loss), {}

    def get_optimizer_param_list(self):
        if self.stage == "gru_training":
            return self.model.parameters()
        else:
            return self.intervention_model.parameters()

    def create_intervention_datasets(self):
        lens = (self.helper.args.var_analysis.max_len, self.helper.args.var_analysis.max_len)
        self.valid_sets.intervention = dataset.RepeatCharDataset(self.helper.args.var_analysis.n_symbols, 1, lens, 5000, seed=1234, max_percent_per_lenght=0.20)
        if lens[1]+1 <= self.helper.args.var_analysis.n_symbols:
            self.valid_sets.intervention_norepeat = dataset.RepeatCharDataset(self.helper.args.var_analysis.n_symbols, 1, lens, 5000, seed=3452, with_repeats=False)
        self.create_loaders()

        self.set_train_set(dataset.RepeatCharDataset(self.helper.args.var_analysis.n_symbols, 1, lens, 1000000, seed=0, exclude=[self.valid_sets.intervention]))

    def set_stage(self, stage: str):
        if self.stage == stage:
            return

        assert stage == "intervention"
        self.stage = stage
        self.create_optimizer()

        self.valid_sets = framework.data_structures.DotDict()
        self.create_intervention_datasets()

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)
        self.helper.saver.register("scaler", self.scaler, replace=True)

    def change_stage(self, is_load: bool = False):
        # Hack: at the loading, add one to the step threshould, because the checkpoint is saved before the switch happens.
        if self.model.training and self.stage == "gru_training" and (self.helper.args.var_analysis.model_train_steps+int(is_load)) <= self.helper.state.iter:
            self.set_stage("intervention")

    def train_step(self):
        self.change_stage()
        return super().train_step()

    def run_model(self, data: Dict[str, torch.Tensor], ubatch: int = 0) -> RecurrentResult:
        if self.stage == "gru_training":
            return self.run_gru_model(data, ubatch)
        else:
            return self.run_model_intervention(data, ubatch)

    def create_intervention(self):
        if self.helper.args.var_analysis.ngram == 1:
            # Faster
            self.intervention = ElementwiseIntervention(self.helper.args.var_analysis.max_len)
        else:
            self.intervention = NgramIntervention(self.helper.args.var_analysis.ngram, len(self.train_set.in_vocabulary), self.helper.args.var_analysis.max_len)

    def create_rnn(self):
        return Model(
            len(self.train_set.in_vocabulary), self.helper.args.state_size,
            self.helper.args.var_analysis.embedding_size, no_input=self.helper.args.var_analysis.no_input)

    def create_model(self):
        self.create_intervention()

        self.n_variables = self.intervention.get_n_variables()

        self.var_size = self.helper.args.var_analysis.var_size
        if self.var_size is None:
            self.var_size = self.helper.args.state_size // self.n_variables

        if self.helper.args.state_size < self.var_size * self.n_variables:
            raise ValueError("Not enough states")

        if self.helper.args.var_analysis.intervention_model == "fixed":
            self.add_model("intervention_model", InterventionModel(
                self.helper.args.state_size, self.var_size, self.n_variables,
                self.helper.args.var_analysis.mask, self.helper.args.var_analysis.mask_reg))
        else:
            # def __init__(self, hidden_size: int, var_size: int, n_variables: int):
            self.add_model("intervention_model", AutosegmentationInterventionModel(
                self.helper.args.state_size, self.var_size, self.n_variables,
                self.helper.args.var_analysis.mask_reg))

        self.stage = "gru_training"

        return self.create_rnn()

