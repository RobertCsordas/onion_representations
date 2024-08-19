import lib

r = lib.get_runs(["rnn_intervention_test_all_decode2", "rnn_intervention_test_all_decode_48"], check_finished=False)
r = [x for x in r if x.state == "finished"]

groups = lib.common.group(r, ["state_size", "decode.n_layers",  "var_analysis.no_input"])
stats = lib.common.calc_stat(groups, lambda k: k in {"validation/valid/accuracy/total"})
print(len(r))
#"validation/intervention/accuracy/total"

sizes = [48, 64, 128, 256, 512, 1024]
names = {1: "Linear", 2: "MLP"}
no_input_names = {0: "Autoregressive", 1: "No input"}

shead = " & ".join(["N="+str(s) for s in sizes])

# print(f" & \\multicolumn{{{len(sizes)}}}{{c}}{{Autoregressive}} & \\multicolumn{{{len(sizes)}}}{{c}}{{Self-scheduled}} \\\\")
print(f"Decoder & Variant & {shead} \\\\")

for nlayers in [1,2]:
    for li, no_input in enumerate([0, 1]):
        line = ""
        for state_size in sizes:
            k = f"state_size_{state_size}/decode.n_layers_{nlayers}/var_analysis.no_input_{no_input}"

            s = stats[k]['validation/valid/accuracy/total'].get()
            line += f" & {s.mean:.2f} $\\pm$ {s.std:.2f}"

        n = f"\\multirow{{2}}{{*}}{{{names[nlayers]}}}" if no_input == 0 else ""
        print(f"{n} & {no_input_names[no_input]} & {line[3:]} \\\\")
    print("\\midrule")

r = lib.get_runs(["rnn_intervention_test_all_decode_godel", "rnn_intervention_test_all_decode_godel_48"], check_finished=False)
r = [x for x in r if x.state == "finished"]

groups = lib.common.group(r, ["state_size", "var_analysis.no_input"])
stats = lib.common.calc_stat(groups, lambda k: k in {"validation/valid/accuracy/total"})
for li, no_input in enumerate([0, 1]):
    line = ""
    for state_size in sizes:
        k = f"state_size_{state_size}/var_analysis.no_input_{no_input}"

        s = stats[k]['validation/valid/accuracy/total'].get()
        line += f" & {s.mean:.2f} $\\pm$ {s.std:.2f}"

    n = f"\\multirow{{2}}{{*}}{{Onion}}" if no_input == 0 else ""
    print(f"{n} & {no_input_names[no_input]} & {line[3:]} \\\\")
print("\\midrule")



r = lib.get_runs(["rnn_intervention_test_all_decode_gru", "rnn_intervention_test_all_decode_gru_48"], check_finished=False)
r = [x for x in r if x.state == "finished"]

groups = lib.common.group(r, ["state_size", "decode.gru.autoregressive", "var_analysis.no_input"])
stats = lib.common.calc_stat(groups, lambda k: k in {"validation/valid/accuracy/total"})
for autoregressive in [1, 0]:
    for li, no_input in enumerate([0, 1]):
        line = ""
        for state_size in sizes:
            k = f"state_size_{state_size}/decode.gru.autoregressive_{autoregressive}/var_analysis.no_input_{no_input}"

            s = stats[k]['validation/valid/accuracy/total'].get()
            line += f" & {s.mean:.2f} $\\pm$ {s.std:.2f}"

        n = f"\\multirow{{2}}{{*}}{{GRU - {'autoregressive' if autoregressive else 'no input'}}}" if no_input == 0 else ""
        print(f"{n} & {no_input_names[no_input]} & {line[3:]} \\\\")
print("\\midrule")
