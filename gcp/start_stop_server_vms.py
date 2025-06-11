import sys
import threading
from pyheaven import CMD

hosts = [f"hothash-node-{i:02d}" for i in range(1, 21)]
zone = "asia-east2-a"
coordinator = "hothash-node-21"


def stop_vm(host):
    CMD(f"bash -c 'source config.bash && gcloud compute instances stop {host} --zone {zone}'"
        )


def start_vm(host):
    CMD(f"bash -c 'source config.bash && gcloud compute instances start {host} --zone {zone}'"
        )


def run_in_threads(action_func):
    threads = []
    for host in hosts + [coordinator]:
        t = threading.Thread(target=action_func, args=(host, ))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("start", "stop"):
        print("Usage: python servers.py [start|stop]")
        sys.exit(1)

    if sys.argv[1] == "start":
        run_in_threads(start_vm)
    else:
        run_in_threads(stop_vm)
