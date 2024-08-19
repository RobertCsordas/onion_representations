import lib

r = lib.get_runs(["rnn_intervention_test_all", "rnn_intervention_test_all_48"], check_finished=False)
r = [x for x in r if x.state == "finished"]

groups = lib.common.group(r, ["state_size", "var_analysis.ngram",  "var_analysis.no_input"])
stats = lib.common.calc_stat(groups, lambda k: k in {"validation/intervention/accuracy/total"})

r = lib.get_runs(["rnn_intervention_test_all_godel", "rnn_intervention_test_all_godel_48"])

groups = lib.common.group(r, ["state_size", "var_analysis.no_input"])
stats_godel = lib.common.calc_stat(groups, lambda k: k in {"validation/intervention_valid/accuracy/total"})

#"validation/intervention/accuracy/total"

sizes = [48, 64, 128, 256, 512, 1024]
names = {1: "Linear Unigram", 2: "Linear Bigram"}
no_input_names = {0: "Autoregressive", 1: "No input"}

shead = " & ".join(["N="+str(s) for s in sizes])

# print(f" & \\multicolumn{{{len(sizes)}}}{{c}}{{Autoregressive}} & \\multicolumn{{{len(sizes)}}}{{c}}{{Self-scheduled}} \\\\")
print(f"Intervention & Variant & {shead} \\\\")

for li, no_input in enumerate([0, 1]):
    print("-----------------------------------")
    print(no_input_names[no_input])
    print("-----------------------------------")
    for ngram in [1,2]:
        line = ""
        for state_size in sizes:
            k = f"state_size_{state_size}/var_analysis.ngram_{ngram}/var_analysis.no_input_{no_input}"

            s = stats[k]['validation/intervention/accuracy/total'].get()
            line += f" & {s.mean:.2f} $\\pm$ {s.std:.2f}"

        # n = f"\\multirow{{2}}{{*}}{{{names[ngram]}}}"# if no_input == 0 else ""
        print(f"{names[ngram]} & {line[3:]} \\\\")


    line = ""
    for state_size in sizes:
        k = f"state_size_{state_size}/var_analysis.no_input_{no_input}"

        s = stats_godel[k]['validation/intervention_valid/accuracy/total'].get()
        line += f" & {s.mean:.2f} $\\pm$ {s.std:.2f}"

    # n = f"\\multirow{{2}}{{*}}{{Onion Unigram}}"# if no_input == 0 else ""
    print(f"Onion Unigram & {line[3:]} \\\\")