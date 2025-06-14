import threading
from pyheaven import CMD

hosts = [f"hothash-node-{i:02d}" for i in range(1, 21)]
zone = "asia-east2-a"


def send_server_to_vm(host):
    CMD(f"gcloud compute scp --zone {zone} ./server.py jiayuan@{host}:~/server.py"
        )


threads = []
for host in hosts:
    threads.append(threading.Thread(target=send_server_to_vm, args=(host, )))
    threads[-1].start()

for t in threads:
    t.join()
print("Send server.py to vms done.")
