from pyheaven import *
import numpy as np

coordinator = ""


def run_exp(identifier, args):
    if ExistFile(f"../result/gcp/profile_{identifier}.json"):
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
        f"source config.bash && gcloud compute scp {coordinator}:~/profile_{identifier}.json ../result/gcp/profile_{identifier}.json"
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
    "spore": "-H spore",
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
    "TD10000": "-D 10000",
    "TD9500": "-D 9500",
    "TD9000": "-D 9000",
    "TD8500": "-D 8500",
    "TD8000": "-D 8000",
    "TD7500": "-D 7500",
    "TD7000": "-D 7000",
    "TD6500": "-D 6500",
    "TD6000": "-D 6000",
    # "SD300": "-D 300",
    # "SD350": "-D 350",
    # "SD400": "-D 400",
    # "SD450": "-D 450",
    # "SD500": "-D 500",
    # "SD550": "-D 550",
    # "SD600": "-D 600",
    # "SD650": "-D 650",
    # "SD700": "-D 700",
    # "D50": "-D 50",
    # "D45": "-D 45",
    # "D40": "-D 40",
    # "D35": "-D 35",
    # "D30": "-D 30",
    # "D25": "-D 25",
    # "D20": "-D 20",
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

GAMMA_ARGS = {
    "G01": "-g 1",
    "G02": "-g 2",
    "G03": "-g 3",
    "G04": "-g 4",
    "G05": "-g 5",
    "G06": "-g 6",
    "G07": "-g 7",
    "G08": "-g 8",
}

THRESH_ARGS = {
    "R01": "-r 1",
    "R02": "-r 2",
    "R03": "-r 3",
    "R04": "-r 4",
    "R05": "-r 5",
    "R06": "-r 6",
    "R07": "-r 7",
}

if __name__ == "__main__":
    # Experiment 1: Varying Workload Skewness
    for n_args, f_args in THRESH_ARGS.items():
        for alg, alg_args in ALG_ARGS.items():
            run_exp(f"{n_args}_{alg}", f"{f_args} {alg_args}")
