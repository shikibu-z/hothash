from pyheaven import *


def run_exp(identifier, args):
    if ExistFile(f"../result/gcp/profile_{identifier}.json"):
        print(f"Skipping experiment '{identifier}'")
        return

    print("=====================================")
    print(f"Running experiment '{identifier}' with args '{args}'")
    command = f"python3 exp.py {args} > profile_{identifier}.json"
    gcloud_command = f"source config.bash && gcloud compute ssh hothash-gke --zone asia-east2-a --command '{command}'"
    CMD(gcloud_command)
    CMD(f"gcloud compute scp hothash-gke:~/profile_{identifier}.json ../result/gcp/profile_{identifier}.json")
    print("=====================================")


ALG_ARGS = {
    "hot": "-H hot",
    "cons": "-H cons",
    "balanced": "-H balanced",
    "bounded": "-H bounded",
    "spore": "-H spore",
}

K8S_ARGS = {
    "D15_K8S": "-D 15 -T 20",
    "D20_K8S": "-D 20 -T 20",
    "D25_K8S": "-D 25 -T 20",
    "D30_K8S": "-D 30 -T 20",
    "D35_K8S": "-D 35 -T 20",
    "D40_K8S": "-D 40 -T 20",
    "D45_K8S": "-D 45 -T 20",
    "D50_K8S": "-D 50 -T 20",
}


if __name__ == "__main__":
    for n_args, f_args in K8S_ARGS.items():
        for alg, alg_args in ALG_ARGS.items():
            run_exp(f"{n_args}_{alg}", f"{f_args} {alg_args}")
