source config.bash
echo "Sending script..."
gcloud compute scp --zone $zone ./03-generate_data_script.py $coordinator:~/data_gen.py
echo "Start generating data..."
gcloud compute ssh $coordinator --zone $zone --command "python3 data_gen.py --num_data 50"
echo "Done."
