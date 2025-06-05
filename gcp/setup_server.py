"""
source config.bash
hosts=(hothash-node-01 hothash-node-02 hothash-node-03 hothash-node-04 hothash-node-05 hothash-node-06 hothash-node-07 hothash-node-08 hothash-node-09 hothash-node-10 hothash-node-11 hothash-node-12 hothash-node-13 hothash-node-14 hothash-node-15 hothash-node-16 hothash-node-17 hothash-node-18 hothash-node-19 hothash-node-20 hothash-node-21)
for host in "${hosts[@]}"
do
    echo "Set up $host..."
    gcloud compute scp --zone $zone ./config.bash $host:~/config.bash
    gcloud compute scp --zone $zone ./config.json $host:~/config.json
    gcloud compute scp --zone $zone ./04-server.py $host:~/server.py
    echo "{\"node_id\": \"$host\"}" > node_id_$host.json
    gcloud compute scp --zone $zone ./node_id_$host.json $host:~/node_id.json
    gcloud compute ssh $host --zone $zone --command "python3 server.py > server.log" &
done

gcloud compute scp --zone $zone ./config.bash $coordinator:~/config.bash
gcloud compute scp --zone $zone ./config.json $coordinator:~/config.json
gcloud compute scp --zone $zone ./06-exp_script.py $coordinator:~/exp.py

sleep 90
"""

import time
import threading
from pyheaven import CMD

hosts = [f"hothash-node-{i:02d}" for i in range(1, 21)]
zone = "asia-east2-a"
coordinator = "hothash-node-21"

def setup_server(host):
    CMD(f"source config.bash && gcloud compute scp --zone {zone} ./config.bash {host}:~/config.bash")
    CMD(f"source config.bash && gcloud compute scp --zone {zone} ./config.json {host}:~/config.json")
    CMD(f"source config.bash && gcloud compute scp --zone {zone} ./04-server.py {host}:~/server.py")
    CMD(f"echo '{{\"node_id\": \"{host}\"}}' > node_id_{host}.json")
    CMD(f"source config.bash && gcloud compute scp --zone {zone} ./node_id_{host}.json {host}:~/node_id.json")
    CMD(f"source config.bash && gcloud compute ssh {host} --zone {zone} --command 'python3 server.py > server.log' &")
def stop_server(host):
    CMD(f"source config.bash && gcloud compute instances stop {host} --zone {zone}")
def start_server(host):
    CMD(f"source config.bash && gcloud compute instances start {host} --zone {zone}")

threads = []
for host in hosts:
    threads.append(threading.Thread(target=stop_server, args=(host,))); threads[-1].start()
threads.append(threading.Thread(target=stop_server, args=(coordinator,))); threads[-1].start()
for t in threads:
    t.join()

time.sleep(120)

threads = []
for host in hosts:
    threads.append(threading.Thread(target=start_server, args=(host,))); threads[-1].start()
threads.append(threading.Thread(target=start_server, args=(coordinator,))); threads[-1].start()
for t in threads:
    t.join()

time.sleep(90)

threads = []
for host in hosts:
    threads.append(threading.Thread(target=setup_server, args=(host,))); threads[-1].start()
CMD(f"source config.bash && gcloud compute scp --zone {zone} ./config.bash {coordinator}:~/config.bash")
CMD(f"source config.bash && gcloud compute scp --zone {zone} ./config.json {coordinator}:~/config.json")
CMD(f"source config.bash && gcloud compute scp --zone {zone} ./exp.py {coordinator}:~/exp.py")
CMD(f"source config.bash && gcloud compute scp --zone {zone} ./dyn_exp.py {coordinator}:~/dyn_exp.py")
CMD(f"source config.bash && gcloud compute scp --zone {zone} ./slide_exp.py {coordinator}:~/slide_exp.py")
CMD(f"source config.bash && gcloud compute scp --zone {zone} ./fix_exp.py {coordinator}:~/fix_exp.py")
for t in threads:
    t.join()
print("Done.")

time.sleep(90)