#!/bin/sh

## declare dataset string arrays
declare -a datasets=("Cora")
# declare -a datasets=("Amazon-ratings" "Tolokers" "Cora_full" "Computers")
# declare -a datasets=("Questions" "Penn94")

for data in "${datasets[@]}"
do

    # python3 main_search.py --lrs_b .01 .001 --lrs_m 1 5 --h_dims 128  --pos_dims 64 128 --num_layers 2 --dropouts 0 .2 --Ks 64 128 --l12s .01 .001 .0001 0 --model_name MLP_LLPE --pos_method llpe --data_name "$data"
    # python3 main_search.py --lrs_b .01 .001 --lrs_m 1 5 --h_dims 128  --pos_dims 64 128 --num_layers 2 --dropouts 0 .2 --Ks 64 128 --l12s .01 .001 .0001 0 --model_name SAGE_LLPE --pos_method llpe --data_name "$data"
    # python3 main_search.py --lrs_b .01 .001 --lrs_m 1 5 --h_dims 128  --pos_dims 64 128 --num_layers 2 --dropouts 0 --lns 0 .00001 --Ks 64 128 --l12s .01 .001 .0001 0 --model_name GT_LLPE --pos_method llpe --data_name "$data"
    
done




