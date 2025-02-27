import json
import numpy as np
from google.cloud import storage
import argparse
from pyheaven import TQDM


def generate_column(data_size=64 * 1024 * 1024, seed=42):
    rng = np.random.RandomState(seed)
    return [rng.randint(0, 100000) for _ in range(data_size)]


def generate_data(num_data=15, bucket_name="hothash-0"):
    config = json.load(open("config.json"))
    client = storage.Client(project=config["project_id"])
    bucket = client.bucket(bucket_name)
    if bucket.exists():
        for blob in bucket.list_blobs():
            blob.delete()
    for i in TQDM(num_data):
        key = f"D{i:03d}"
        value = generate_column(seed=i)
        blob = bucket.blob(f"{key}.json")
        blob.upload_from_string(json.dumps(value), timeout=1000.0)
        del value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_data", type=int, default=15)
    args = parser.parse_args()
    generate_data(num_data=args.num_data)
