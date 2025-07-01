import time
import json
from google.cloud import storage
from collections import OrderedDict
from flask import Flask, request, jsonify
from threading import Lock

import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

app = Flask(__name__)
CACHE = OrderedDict()
NUM_QUERIES = 0
NUM_REPLICAS = 0
TIMEOUT = 5
config = json.load(open("config.json"))
K = config["node_lru_cache_size"]
client = storage.Client(project=config["project_id"])
bucket = client.bucket(config["bucket_name"])
node_id = json.load(open("node_id.json"))["node_id"]
node_internal_ip = config["node_internal_ips"][node_id]
lock = Lock()


def fetch_data(key):
    global NUM_REPLICAS
    NUM_REPLICAS += 1
    for _ in range(TIMEOUT):
        try:
            return json.loads(bucket.blob(f"{key}.json").download_as_string())
        except:
            pass
    return list()


def compute(op, data):
    if op == "sum":
        return sum(data)
    elif op == "mean":
        return sum(data) // len(data)
    elif op == "min":
        return min(data)
    elif op == "max":
        return max(data)
    elif op == "mix":
        return sum(data)
        # return sum(data) + min(data) + max(data)
    elif op == "sort":
        return sum(sorted(data)[: len(data) // 2])
    else:
        return None


@app.route("/info")
def info():
    global CACHE
    global NUM_QUERIES
    global K
    version = "B"
    return jsonify(
        {
            "version": version,
            "num_queries": NUM_QUERIES,
            "num_replicas": NUM_REPLICAS,
            "cache_size": K,
            "cached_keys": {key: True for key in CACHE.keys()},
        }
    )


@app.route("/query")
def query():
    global CACHE
    global NUM_QUERIES
    NUM_QUERIES += 1
    op = str(request.args.get("op"))
    data_id = str(request.args.get("data_id"))
    query_id = str(request.args.get("query_id"))
    start_time = time.time()
    cache_miss = 1
    cache_replacement = 0

    for _ in range(TIMEOUT):
        try:
            with lock:
                if data_id in CACHE:
                    data = CACHE[data_id]
                    CACHE.move_to_end(data_id)
                    cache_miss = 0
                else:
                    data = None
                    while data is None:
                        data = fetch_data(data_id)
                    if K > 0 and len(CACHE) >= K:
                        print(node_id, "pop")
                        CACHE.popitem(last=False)
                        cache_replacement = 1
                    CACHE[data_id] = data
                    CACHE.move_to_end(data_id)
            fetch_time = time.time()

            if op == "init_cache":
                result = {}
                break

            result = int(compute(op, data))
            break
        except Exception as e:
            print(e)
            fetch_time = time.time()
            result = 0
    end_time = time.time()
    return jsonify(
        {
            "result": result,
            "tot_server_time": end_time - start_time,
            "fetch_time": fetch_time - start_time,
            "compute_time": end_time - fetch_time,
            "data_id": data_id,
            "node_id": node_id,
            "query_id": query_id,
            "query_op": op,
            "cache_miss": cache_miss,
            "cache_replacement": cache_replacement,
        }
    )


@app.route("/clear_cache")
def clear_cache():
    global CACHE
    global NUM_QUERIES
    global NUM_REPLICAS
    with lock:
        CACHE.clear()
        NUM_QUERIES = 0
        NUM_REPLICAS = 0
    return jsonify({"status": "success"})


def shutdown_server():
    func = request.environ.get("werkzeug.server.shutdown")
    if func is None:
        raise RuntimeError("Not running with the Werkzeug Server")
    func()


@app.route("/stop")
def stop():
    shutdown_server()
    return "Server shutting down..."


def start_server(host, port=8080):
    app.run(debug=True, host=host, port=port, threaded=True)
    # subprocess.call(["gunicorn", "server:app", "-b", f"{host}:{port}", "--daemon"])


if __name__ == "__main__":
    start_server(host=node_internal_ip, port=8080)
