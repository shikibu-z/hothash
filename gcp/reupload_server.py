import time
import threading
from pyheaven import CMD

hosts = [f"hothash-node-{i:02d}" for i in range(1, 21)]
zone = "asia-east2-a"


def setup_server(host):
    CMD(f"gcloud compute ssh jiayuan@{host} --zone {zone} --command 'pkill -f server.py'"
        )
    CMD(f"gcloud compute scp --zone {zone} ./server.py jiayuan@{host}:~/server.py"
        )
    CMD(f"gcloud compute ssh jiayuan@{host} --zone {zone} --command 'python3 server.py > server.log' &"
        )


threads = []
for host in hosts:
    threads.append(threading.Thread(target=setup_server, args=(host, )))
    threads[-1].start()

for t in threads:
    t.join()
print("Done.")

time.sleep(5)
