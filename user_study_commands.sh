./run_experiment.sh p3 break_room.wbt 1

./run_experiment.sh p3 break_room.wbt 2
./run_clustering.sh p3 break_room.wbt 2 --eps 0.2 --min-samples 15 --start-row 35 --step 10 --image-size 200 200

./run_experiment.sh p3 break_room.wbt 3
./run_clustering.sh p3 break_room.wbt 3 --eps 0.2 --min-samples 15 --start-row 35 --step 10 --image-size 200 200

./run_experiment.sh p3 hall.wbt 1

./run_experiment.sh p3 hall.wbt 2
./run_clustering.sh p3 hall.wbt 2  --eps 0.2 --min-samples 15 --start-row 35 --step 10 --image-size 200 200

./run_experiment.sh p3 hall.wbt 3
./run_clustering.sh p3 hall.wbt 3  --eps 0.2 --min-samples 15 --start-row 35 --step 10 --image-size 200 200

./run_experiment.sh p3 apartment.wbt 1

./run_experiment.sh p3 apartment.wbt 2
./run_clustering.sh p3 apartment.wbt 2 --eps 0.2 --min-samples 15 --start-row 35 --step 10 --image-size 200 200

./run_experiment.sh p3 apartment.wbt 3
./run_clustering.sh p3 apartment.wbt 3 --eps 0.2 --min-samples 15 --start-row 35 --step 10 --image-size 200 200
