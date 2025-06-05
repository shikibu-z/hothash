import time
import threading
from pyheaven import *


def run_exp(identifier, args):
    if ExistFile(f"../result/gcp/profile_{identifier}.json"):
        print(f"Skipping experiment '{identifier}'")
        return

    print("=====================================")
    print(f"Running experiment '{identifier}' with args '{args}'")
    command = f"python3 fix_exp.py {args} > profile_{identifier}.json"
    gcloud_command = f"source config.bash && gcloud compute ssh hothash-node-21 --zone asia-east2-a --command '{command}'"
    CMD(gcloud_command)
    CMD(f"gcloud compute scp hothash-node-21:~/profile_{identifier}.json ../result/gcp/profile_{identifier}.json")
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
    # "cons": "-H cons",
    # "balanced": "-H balanced",
    # "bounded": "-H bounded",
    # "spore": "-H spore",
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

CHURN_ARGS = {
    "C01": "-C 1",
    "C02": "-C 2",
    "C03": "-C 3",
    "C04": "-C 4",
    "C05": "-C 5",
}

RANDOM_ARGS = {
    "R01": "-R 1",  # 15 verification
    "R02": "-R 2",  # 8
    "R03": "-R 3",  # 5
    "R04": "-R 4",  # 4
    "R05": "-R 5",  # 3
    "R06": "-R 6",  # 3
    "R07": "-R 7",  # 3
    # "R08": "-R 8",  # 2
    # "R09": "-R 9",  # 2
    # "R10": "-R 10",  # 2
    "R15": "-R 15",  # 1
}

OP_ARGS = {
    "DYNOP_CP_D15": "-D 15 -O True",
    "DYNOP_CP_D20": "-D 20 -O True",
    "DYNOP_CP_D25": "-D 25 -O True",
    "DYNOP_CP_D30": "-D 30 -O True",
    "DYNOP_CP_D35": "-D 35 -O True",
    "DYNOP_CP_D40": "-D 40 -O True",
    "DYNOP_CP_D45": "-D 45 -O True",
    "DYNOP_CP_D50": "-D 50 -O True",
    # "DYNOP_D15": "-D 15",
    # "DYNOP_D20": "-D 20",
    # "DYNOP_D25": "-D 25",
    # "DYNOP_D30": "-D 30",
    # "DYNOP_D35": "-D 35",
    # "DYNOP_D40": "-D 40",
    # "DYNOP_D45": "-D 45",
    # "DYNOP_D50": "-D 50",
}

SLIDE_ARGS = {
    "SLIDE_D15": "-D 15",
    "SLIDE_D20": "-D 20",
    "SLIDE_D25": "-D 25",
    "SLIDE_D30": "-D 30",
    "SLIDE_D35": "-D 35",
    "SLIDE_D40": "-D 40",
    "SLIDE_D45": "-D 45",
    "SLIDE_D50": "-D 50",
}

# default hothash with hotness change
HOTNESS_CHANGE_ARGS = {
    "HC00": "-X 0 -k 1.5",
    "HC01": "-X 1 -k 1.5",
    "HC02": "-X 2 -k 1.5",
    "HC03": "-X 3 -k 1.5",
    "HC04": "-X 4 -k 1.5",
    "HC05": "-X 5 -k 1.5",
}

# hothash with sliding window and hotness change
HOTNESS_CHANGE_WINDOW_ARGS = {
    "HCW00": "-W 1 -X 0 -k 1.5",
    "HCW01": "-W 1 -X 1 -k 1.5",
    "HCW02": "-W 1 -X 2 -k 1.5",
    "HCW03": "-W 1 -X 3 -k 1.5",
    "HCW04": "-W 1 -X 4 -k 1.5",
    "HCW05": "-W 1 -X 5 -k 1.5",
}

# hothash with window size change, hotness change 5 times
WINDOW_SIZE_ARGS = {
    "WS00": "-X 3 -k 1.5 -W 0",
    "WS01": "-X 3 -k 1.5 -W 1",
    "WS02": "-X 3 -k 1.5 -W 2",
    "WS03": "-X 3 -k 1.5 -W 3",
    "WS04": "-X 3 -k 1.5 -W 4",
    "WS05": "-X 3 -k 1.5 -W 5",
}

K8S_ARGS = {
    "D15_K8S": "-D 15",
    "D20_K8S": "-D 20",
    "D25_K8S": "-D 25",
    "D30_K8S": "-D 30",
    "D35_K8S": "-D 35",
    "D40_K8S": "-D 40",
    "D45_K8S": "-D 45",
    "D50_K8S": "-D 50",
}

FIX_WINDOW_ARGS = {
    "FIX00": "-T 40 -W 1 -X 0 -k 1.5",
    "FIX01": "-T 50 -W 1 -X 1 -k 1.5",
    "FIX02": "-T 60 -W 1 -X 2 -k 1.5",
    "FIX03": "-T 60 -W 1 -X 3 -k 1.5",
    "FIX04": "-T 75 -W 1 -X 4 -k 1.5",
    "FIX05": "-T 90 -W 1 -X 5 -k 1.5",
}


zone = "asia-east2-a"
coordinator = "hothash-node-21"
hosts = [f"hothash-node-{i:02d}" for i in range(1, 21)]


def stop_server(host):
    CMD(f"source config.bash && gcloud compute instances stop {host} --zone {zone}")


if __name__ == "__main__":
    # for n_args, f_args in HOTNESS_CHANGE_ARGS.items():
    #     for alg, alg_args in ALG_ARGS.items():
    #         run_exp(f"{n_args}_{alg}", f"{f_args} {alg_args}")

    for n_args, f_args in FIX_WINDOW_ARGS.items():
        for alg, alg_args in ALG_ARGS.items():
            run_exp(f"{n_args}_{alg}", f"{f_args} {alg_args}")

    # for n_args, f_args in WINDOW_SIZE_ARGS.items():
    #     for alg, alg_args in ALG_ARGS.items():
    #         run_exp(f"{n_args}_{alg}", f"{f_args} {alg_args}")

    threads = []
    for host in hosts:
        threads.append(threading.Thread(target=stop_server, args=(host,)))
        threads[-1].start()
    threads.append(threading.Thread(target=stop_server, args=(coordinator,)))
    threads[-1].start()
    for t in threads:
        t.join()
    time.sleep(120)
