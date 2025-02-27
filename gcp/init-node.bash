hosts=(node-name)
for host in "${hosts[@]}"
do
    echo "Initializing $host..."
    gcloud compute ssh $host --zone $zone --command "
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
    python3 -m pip install -U numpy<2.0.0
    python3 -m pip install -U pyheaven
    python3 -m pip install -U tqdm
"
