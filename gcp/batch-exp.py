from pyheaven import *
import numpy as np

coordinator = ""


def run_exp(identifier, args):
    if ExistFile(f"results/profile_{identifier}.json"):
        print(f"Skipping experiment '{identifier}'")
        return

    print("=====================================")
    print(f"Running experiment '{identifier}' with args '{args}'")
    command = f"python3 exp.py {args} > profile_{identifier}.json"
    gcloud_command = (
        f"source config.bash && gcloud compute ssh {coordinator} --command '{command}'"
    )
    CMD(gcloud_command)
    CMD(
        f"source config.bash && gcloud compute scp {coordinator}:~/profile_{identifier}.json ./results/profile_{identifier}.json"
    )
    print("=====================================")


EMPTY = [
    {
        "avg_duration_secs": 0,
        "p99_latency": 0,
    },
    {},
]

ALG_ARGS = {
    "hot": "-H hot",
    "cons": "-H cons",
    "balanced": "-H balanced",
    "bounded": "-H bounded",
}

SKEWNESS_ARGS = {
    "Y50": "-M ycsb -k 1.5",
    "Y30": "-M ycsb -k 1.3",
    "Y40": "-M ycsb -k 1.4",
    "Y20": "-M ycsb -k 1.2",
    "Y10": "-M ycsb -k 1.1",
    "Y01": "-M ycsb -k 1.01",
    "Y00": "-M uniform",
    "S50": "-M  ssb -k 1.5",
    "S40": "-M  ssb -k 1.4",
    "S30": "-M  ssb -k 1.3",
    "S20": "-M  ssb -k 1.2",
    "S10": "-M  ssb -k 1.1",
    "S01": "-M  ssb -k 1.01",
}

DATASET_SIZE_ARGS = {
    "D50": "-D 50",
    "D45": "-D 45",
    "D40": "-D 40",
    "D35": "-D 35",
    "D30": "-D 30",
    "D25": "-D 25",
    "D20": "-D 20",
    # "D15": "-D 15",
}

NUM_QUERIES_ARGS = {
    "Q300": "-q 300",
    "Q400": "-q 400",
    # "Q500": "-q 500",
    "Q600": "-q 600",
    "Q700": "-q 700",
    "Q800": "-q 800",
}

EPSILON_ARGS = {
    "E10": "-e 1.0",
    "E09": "-e 0.9",
    "E08": "-e 0.8",
    "E07": "-e 0.7",
    "E06": "-e 0.6",
    "E05": "-e 0.5",
    "E04": "-e 0.4",
    # "E03": "-e 0.3",
    "E02": "-e 0.2",
    "E01": "-e 0.1",
}

ALPHA_ARGS = {
    # "A10": "-a 1.0",
    "A09": "-a 0.9",
    "A11": "-a 1.1",
    "A08": "-a 0.8",
    "A12": "-a 1.2",
    "A05": "-a 0.5",
    "A15": "-a 1.5",
}


def Compute(json):
    return json[0]["avg_duration_secs"], json[0]["p99_latency"]


import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


def Plot(fig_path, xlabel, xticks, data):
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 18
    plt.rcParams["legend.framealpha"] = 0

    plt.rcParams["figure.figsize"] = (15, 5)

    plt.rcParams["lines.linewidth"] = 3
    plt.rcParams["lines.markersize"] = 10
    plt.rcParams["legend.edgecolor"] = "none"

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set(xlabel=xlabel)
    ax1.set_xticklabels(xticks)
    ax1.set(ylabel="Execution Time (sec)")
    ax2.set(xlabel=xlabel)
    ax2.set_xticklabels(xticks)
    ax2.set(ylabel="P99 Latency (sec)")

    for (ls, marker), (alg, alg_data) in zip(
        [("--", "s"), (":", "^"), ("-", "o"), ("-.", "X")], data.items()
    ):
        jsons = list(alg_data.values())
        stats = [Compute(json) for json in jsons]
        ax1.plot(xticks, [s[0] for s in stats], ls=ls, marker=marker)
        ax2.plot(xticks, [np.log(s[1]) for s in stats], ls=ls, marker=marker)

    ax1.yaxis.set_minor_formatter(FormatStrFormatter("%d"))
    ax1.tick_params(axis="y", which="minor", labelsize=12, colors="grey")
    ax1.grid(axis="y", linestyle="--")
    ax2.yaxis.set_minor_formatter(FormatStrFormatter("%d"))
    ax2.tick_params(axis="y", which="minor", labelsize=12, colors="grey")
    ax2.grid(axis="y", linestyle="--")

    legend = list(data.keys())
    fig.legend(legend, loc="upper center", ncol=3)
    fig.savefig(f"{fig_path}.pdf", bbox_inches="tight")


def bundled(result, B=10):
    profile, records = result
    n_group = len(records) // B
    assignment = [i // B for i in range(len(records))]
    np.random.seed(42)
    np.random.shuffle(assignment)
    groups = {i: [] for i in range(n_group)}
    for i, (qid, info) in zip(assignment, records.items()):
        groups[i].append((qid, info))

    latencies = [
        max(info["local_time"] for qid, info in group) for group in groups.values()
    ]
    avg_duration_secs = np.mean(latencies)
    P99_latency = np.percentile(latencies, 99.0)
    return [{"avg_duration_secs": avg_duration_secs, "p99_latency": P99_latency}, {}]


if __name__ == "__main__":
    args = HeavenArguments.from_parser(
        [
            SwitchArgumentDescriptor("plot-only", help="Only plot the results"),
        ]
    )
    if not args.plot_only:
        CMD("python3 04-setup_server.py")

    # Experiment 1: Varying Workload Skewness
    if not args.plot_only:
        for skewness, f_args in SKEWNESS_ARGS.items():
            for alg, alg_args in ALG_ARGS.items():
                run_exp(f"{skewness}_{alg}", f"{f_args} {alg_args}")
    Plot(
        fig_path="figures/Skewness-YCSB",
        xlabel="Skewness Factor",
        xticks=["uniform", "1.01", "1.1", "1.2", "1.3", "1.4", "1.5"],
        data={
            k: {
                x: (
                    LoadJson(f"./results/profile_{x}_{k}.json")
                    if ExistFile(f"./results/profile_{x}_{k}.json")
                    else EMPTY
                )
                for x in ["Y00", "Y01", "Y10", "Y20", "Y30", "Y40", "Y50"]
            }
            for k in ["balanced", "hot", "cons"]
        },
    )
    Plot(
        fig_path="figures/Skewness-SSB",
        xlabel="Skewness Factor",
        xticks=["uniform", "1.01", "1.1", "1.2", "1.3", "1.4", "1.5"],
        data={
            k: {
                x: (
                    LoadJson(f"./results/profile_{x}_{k}.json")
                    if ExistFile(f"./results/profile_{x}_{k}.json")
                    else EMPTY
                )
                for x in ["Y00", "S01", "S10", "S20", "S30", "S40", "S50"]
            }
            for k in ["bounded", "balanced", "hot", "cons"]
        },
    )
    Plot(
        fig_path="figures/Skewness-YCSB-10",
        xlabel="Skewness Factor",
        xticks=["uniform", "1.01", "1.1", "1.2", "1.3", "1.4", "1.5"],
        data={
            k: {
                x: (
                    bundled(LoadJson(f"./results/profile_{x}_{k}.json"))
                    if ExistFile(f"./results/profile_{x}_{k}.json")
                    else EMPTY
                )
                for x in ["Y00", "Y01", "Y10", "Y20", "Y30", "Y40", "Y50"]
            }
            for k in ["bounded", "balanced", "hot", "cons"]
        },
    )
    Plot(
        fig_path="figures/Skewness-SSB-10",
        xlabel="Skewness Factor",
        xticks=["uniform", "1.01", "1.1", "1.2", "1.3", "1.4", "1.5"],
        data={
            k: {
                x: (
                    bundled(LoadJson(f"./results/profile_{x}_{k}.json"))
                    if ExistFile(f"./results/profile_{x}_{k}.json")
                    else EMPTY
                )
                for x in ["Y00", "S01", "S10", "S20", "S30", "S40", "S50"]
            }
            for k in ["bounded", "balanced", "hot", "cons"]
        },
    )

    # Experiment 0: Varying Alpha
    # if not args.plot_only:
    #     for alpha, alpha_args in ALPHA_ARGS.items():
    #         run_exp(f"{alpha}_hot", f"-H hot {alpha_args}")
    #     Plot(
    #         fig_path = "figures/Alpha",
    #         xlabel = "Alpha",
    #         xticks = ["0.5", "0.8", "0.9", "1.0", "1.1", "1.2", "1.5"],
    #         data = {
    #             "hot": {
    #                 x: LoadJson(f"./results/profile_{x}_hot.json")[0] if ExistFile(f"./results/profile_{x}_hot.json") else EMPTY
    #                 for x in ["A05", "A08", "A09", "A10", "A11", "A12", "A15"]
    #             }
    #         }
    #     )

    # Experiment 2: Varying Dataset Size
    if not args.plot_only:
        for dataset_size, f_args in DATASET_SIZE_ARGS.items():
            for alg, alg_args in ALG_ARGS.items():
                run_exp(f"{dataset_size}_{alg}", f"{f_args} {alg_args}")
    # Plot(
    #     fig_path = "figures/Datasize",
    #     xlabel = "Data Size (x64M)",
    #     xticks = ["15", "20", "25", "30", "35", "40", "45", "50"],
    #     data = {
    #         k: {
    #             x: LoadJson(f"./results/profile_{x}_{k}.json")[0] if ExistFile(f"./results/profile_{x}_{k}.json") else EMPTY
    #             for x in ["D15", "D20", "D25", "D30", "D35", "D40", "D45", "D50"]
    #         } for k in ['bounded', 'balanced', 'hot', 'cons']
    #     }
    # )

    # Experiment 3: Varying Number of Queries
    if not args.plot_only:
        for num_queries, f_args in NUM_QUERIES_ARGS.items():
            for alg, alg_args in ALG_ARGS.items():
                run_exp(f"{num_queries}_{alg}", f"{f_args} {alg_args}")
#     Plot(
#         fig_path = "figures/Workload",
#         xlabel = "Number of Queries per Batch",
#         xticks = ["300", "400", "500", "600", "700", "800"],
#         data = {
#             k: {
#                 x: LoadJson(f"./results/profile_{x}_{k}.json")[0] if ExistFile(f"./results/profile_{x}_{k}.json") else EMPTY
#                 for x in ["Q300", "Q400", "Q500", "Q600", "Q700", "Q800"]
#             } for k in ['bounded', 'balanced', 'hot', 'cons']
#         }
#     )

# Experiment 4: Varying Epsilon
# if not args.plot_only:
#     for epsilon, args in EPSILON_ARGS.items():
#         for alg, alg_args in ALG_ARGS.items():
#             run_exp(f"{epsilon}_{alg}", f"{args} {alg_args}")
#     Plot(
#         fig_path = "figures/Epsilon",
#         xlabel = "Epsilon",
#         xticks = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"],
#         data = {
#             k: {
#                 x: LoadJson(f"./results/profile_{x}_{k}.json")[0] if ExistFile(f"./results/profile_{x}_{k}.json") else EMPTY
#                 for x in ["E01", "E02", "E03", "E04", "E05", "E06", "E07", "E08", "E09", "E10"]
#             } for k in ['bounded', 'balanced', 'hot', 'cons']
#         }
#     )
