import lib
from collections import OrderedDict

results = OrderedDict()
results["IID"] = "validation/valid/accuracy/total"
# results["+1"] = "validation/valid_longer_1/accuracy/total"
# results["+2"] = "validation/valid_longer_2/accuracy/total"

r = lib.get_runs(["rnn_intervention_test_all", "rnn_intervention_test_all_48"], check_finished=False)
r = [x for x in r if x.state == "finished" and x.config["var_analysis.ngram"] == 1]

groups = lib.common.group(r, ["state_size", "var_analysis.no_input"])
stats = lib.common.calc_stat(groups, lambda k: k in results.values())
print(len(r))
#"validation/intervention/accuracy/total"

sizes = [48, 64, 128, 256, 512, 1024]
names = {1: "Unigram", 2: "Bigram"}
no_input_names = {0: "Autoregressive", 1: "No input"}

shead = " & ".join(["N="+str(s) for s in sizes])

# print(f" & \\multicolumn{{{len(sizes)}}}{{c}}{{Autoregressive}} & \\multicolumn{{{len(sizes)}}}{{c}}{{Self-scheduled}} \\\\")
print(f"Intervention & Variant & {shead} \\\\")

for resn, resk in results.items():
    for li, no_input in enumerate([0, 1]):
        line = ""
        for state_size in sizes:
            k = f"state_size_{state_size}/var_analysis.no_input_{no_input}"

            s = stats[k][resk].get()
            line += f" & {s.mean:.2f} $\\pm$ {s.std:.2f}"

        # n = f"\\multirow{{2}}{{*}}{{{resn}}}" if no_input == 0 else ""
        # print(f"{n} & {no_input_names[no_input]} & {line[3:]} \\\\")
        print(f"{no_input_names[no_input]} & {line[3:]} \\\\")
    print("\\midrule")
