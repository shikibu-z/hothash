import threading
from pyheaven import CMD

hosts = [f"hothash-node-{i:02d}" for i in range(1, 21)]
zone = "asia-east2-a"
coordinator = "hothash-node-21"


def setup_server(host):
    CMD(f"gcloud compute scp --zone {zone} ./config.bash shikibu@{host}:~/config.bash")
    CMD(f"gcloud compute scp --zone {zone} ./config.json shikibu@{host}:~/config.json")
    CMD(f"gcloud compute scp --zone {zone} ./server.py shikibu@{host}:~/server.py")
    CMD(f'echo \'{{"node_id": "{host}"}}\' > node_id_{host}.json')
    CMD(
        f"gcloud compute scp --zone {zone} ./node_id_{host}.json shikibu@{host}:~/node_id.json"
    )
    CMD(
        f"gcloud compute ssh shikibu@{host} --zone {zone} --command 'python3 server.py > server.log' &"
    )


threads = []
for host in hosts:
    threads.append(threading.Thread(target=setup_server, args=(host,)))
    threads[-1].start()
for t in threads:
    t.join()

CMD(
    f"gcloud compute scp --zone {zone} ./config.bash shikibu@{coordinator}:~/config.bash"
)
CMD(
    f"gcloud compute scp --zone {zone} ./config.json shikibu@{coordinator}:~/config.json"
)
CMD(f"gcloud compute scp --zone {zone} ./exp.py shikibu@{coordinator}:~/exp.py")
CMD(f"gcloud compute scp --zone {zone} ./dyn_exp.py shikibu@{coordinator}:~/dyn_exp.py")
CMD(
    f"gcloud compute scp --zone {zone} ./slide_exp.py shikibu@{coordinator}:~/slide_exp.py"
)
CMD(f"gcloud compute scp --zone {zone} ./fix_exp.py shikibu@{coordinator}:~/fix_exp.py")
CMD(f"gcloud compute scp --zone {zone} ./rl.py shikibu@{coordinator}:~/rl.py")
CMD(
    f"gcloud compute scp --zone {zone} ./rl_train.py shikibu@{coordinator}:~/rl_train.py"
)
CMD(
    f"gcloud compute scp --zone {zone} ./extract_rl_metric.py shikibu@{coordinator}:~/extract_rl_metric.py"
)
