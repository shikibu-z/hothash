import sys
import threading
from pyheaven import CMD

hosts = [f"hothash-node-{i:02d}" for i in range(1, 21)]
zone = "asia-east2-a"


def start_server(host):
    CMD(
        f"gcloud compute ssh shikibu@{host} --zone {zone} "
        f"--command 'nohup python3 server.py > server.log 2>&1 &'"
    )


def stop_server(host):
    CMD(
        f"gcloud compute ssh shikibu@{host} --zone {zone} "
        f"--command \"pkill -f 'python3 server.py'\""
    )


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in {"start", "stop"}:
        print("Usage: python start_servers.py [start|stop]")
        sys.exit(1)

    action = sys.argv[1]
    threads = []

    for host in hosts:
        t = threading.Thread(
            target=start_server if action == "start" else stop_server, args=(host,)
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print(f"All servers {'started' if action == 'start' else 'stopped'}.")


if __name__ == "__main__":
    main()
