import lib
import matplotlib.pyplot as plt

def get_runs(sweeps):
    r = lib.get_runs(sweeps, check_finished=False)
    return [x for x in r if x.state == "finished" and  x.config["state_size"] == 64 and x.config["var_analysis.no_input"] == 0]


r = get_runs(["rnn_intervention_test_all_decode"])

groups = lib.common.group(r, ["decode.n_layers"])
stats = lib.common.calc_stat(groups, lambda k: k in {"validation/valid/accuracy/total"})
stats = {f"mlp/{k}": v for k, v in stats.items()}


r = get_runs(["rnn_intervention_test_all_decode_godel"])
s = lib.common.calc_stat({"all": r}, lambda k: k in {"validation/valid/accuracy/total"})
stats.update({f"godel/{k}": v for k, v in s.items()})

r = get_runs(["rnn_intervention_test_all_decode_gru"])
groups = lib.common.group(r, ["decode.gru.autoregressive"])
s = lib.common.calc_stat(groups, lambda k: k in {"validation/valid/accuracy/total"})

stats.update({f"gru/{k}": v for k, v in s.items()})

stats = {k: v["validation/valid/accuracy/total"].get() for k, v in stats.items()}

print(list(stats.keys()))

variants = [
    ("Linear", "mlp/decode.n_layers_1"),
    ("MLP", "mlp/decode.n_layers_2"),
    ("Onion", "godel/all"),
    ("GRU - AR.", "gru/decode.gru.autoregressive_1"),
    ("GRU - No in.", "gru/decode.gru.autoregressive_0"),
]

fig, ax=plt.subplots(figsize=(4, 1.5))
plt.bar(range(len(variants)), [stats[v[1]].mean * 100 for v in variants], yerr=[stats[v[1]].std * 100 for v in variants])
plt.xticks(range(len(variants)), [v[0] for v in variants], fontsize=8)
plt.xlabel("Probe")
plt.ylabel("Accuracy [\\%]")
plt.tight_layout()
plt.savefig("probe_godel.pdf", bbox_inches='tight')
# ax.set_yscale('log')
