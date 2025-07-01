source config.bash
hosts=(hothash-node-01 hothash-node-02 hothash-node-03 hothash-node-04 hothash-node-05 hothash-node-06 hothash-node-07 hothash-node-08 hothash-node-09 hothash-node-10 hothash-node-11 hothash-node-12 hothash-node-13 hothash-node-14 hothash-node-15 hothash-node-16 hothash-node-17 hothash-node-18 hothash-node-19 hothash-node-20 hothash-node-21)

for host in "${hosts[@]}"
do
    echo "Initializing $host..."
    gcloud compute ssh shikibu@$host --zone $zone --command "
        sudo apt-get update
        sudo apt-get install python3 python3-pip -y
        python3 -m pip install -U flask
        python3 -m pip install -U jsonlines
        python3 -m pip install -U google-cloud
        python3 -m pip install -U google-cloud-storage
        python3 -m pip install -U google-api-python-client
        python3 -m pip install -U gunicorn
    "
done

gcloud compute ssh $coordinator --zone $zone --command "
    python3 -m pip install -U numpy
    python3 -m pip install -U pyheaven
    python3 -m pip install -U tqdm
    python3 -m pip install -U torch
    python3 -m pip install -U matplotlib
    python3 -m pip install -U openai
"
