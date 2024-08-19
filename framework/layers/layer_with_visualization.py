import torch
import torch.nn
from typing import Dict, Any, Union, List
from ..data_structures import DotDict


class LayerWithVisualization:
    def __init__(self):
        self.visualization_enabled = False

    def prepare(self):
        # Should be called before the training step
        pass

    def plot(self, options: DotDict) -> Dict[str, Any]:
        return {}


class LayerVisualizer:
    def __init__(self, modules: Union[torch.nn.Module, List[torch.nn.Module]], options: Dict[str, Any] = {}):
        self.modules = []
        self.options = options
        self.curr_options = None

        if isinstance(modules, torch.nn.Module):
            modules = [modules]

        for module in modules:
            for n, m in module.named_modules():
                if isinstance(m, LayerWithVisualization):
                    self.modules.append((n, m))

    def set_options(self, options: Dict[str, Any]):
        self.options = options

    def plot(self) -> Dict[str, Any]:
        res = {}
        for n, m in self.modules:
            res.update({f"{n}/{k}": v for k, v in m.plot(self.curr_options).items()})
            m.visualization_enabled = False

        self.curr_options = None
        return res

    def prepare(self, options: Dict[str, Any] = {}):
        self.curr_options = DotDict(self.options)
        self.curr_options.update(options)

        for _, m in self.modules:
            m.prepare()
            m.visualization_enabled = True
