#!/bin/sh

# activate the correct conda environment ~ pytorch (torch)
conda activate env

## declare gnn string arrays
# declare -a mpnn_gnns=("MLP" "GCN")
# declare -a gt_gnns=("MLP" "GCN" "GT" "GPS" "HGT" "HGTX")

## declare dataset string arrays
declare -a datasets=("Amazon-ratings" "Tolokers" "Cora_full" "Computers")
# declare -a datasets=("Questions" "Penn94")

## launch hyperparameter search for each mp-gnn and dataset 
# for gnn in "${mpnn_gnns[@]}"
# do
#     for data in "${datasets[@]}"
#     do
#         python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --pos_dims 1 --h_dims 64 --num_layers 1 2 3 --model_name "$gnn" --pos_method none --data_name "$data"
#     done
# done

## launch hyperparameter search for each gt-gnn and dataset 
# for gnn in "${gt_gnns[@]}"
# do
#     for data in "${datasets[@]}"
#     do
#         python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --pos_dims 2 4 6 8 10 --h_dims 64 --num_layers 1 2 3 --model_name "$gnn" --pos_method lap-base --data_name "$data"
#     done
# done

for data in "${datasets[@]}"
do
    
    # python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --pos_dims 1 --h_dims 64 128 --num_layers 2 --dropouts 0 .2 --model_name MLP --pos_method none --data_name "$data"
    # python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --pos_dims 1 --h_dims 64 128 --num_layers 2 --dropouts 0 .2 --model_name MLP --pos_method elastic --data_name "$data"
    # python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --pos_dims 8 16 24 --h_dims 64 128 --num_layers 2 --dropouts 0 .2 --model_name MLP --pos_method lap-fk --data_name "$data"
    # python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --pos_dims 8 16 24 --h_dims 64 128 --num_layers 2 --dropouts 0 .2 --model_name MLP --pos_method lap-flk --data_name "$data"
    # python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --pos_dims 1 --h_dims 64 128 --num_layers 2 --dropouts 0 .2 --model_name MLP --pos_method lap-full --data_name "$data"
    # python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --pos_dims 8 16 24 --h_dims 64 128 --num_layers 2 --dropouts 0 .2 --model_name MLP --pos_method rwse --data_name "$data"
    # python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --h_dims 64 128  --pos_dims 64 128 --num_layers 2 --dropouts 0 .2 --Ks 64 128 --l12s .001 .0001 --model_name MLP_LLPE --pos_method llpe --data_name "$data"

    # python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --h_dims 64 128  --pos_dims 64 128 256 --num_layers 2 --dropouts 0 .2 --Ks 64 128 256 --l12s .001 .0001 0 --model_name GCN_LLPE --pos_method llpe --data_name "$data"
    # python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --pos_dims 1 --h_dims 64 128 --num_layers 2 --dropouts 0 .2 --model_name GCN --pos_method none --data_name "$data"
    # python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --pos_dims 8 16 24 --h_dims 64 128 --num_layers 2 --dropouts 0 .2 --model_name GCN --pos_method lap-fk --data_name "$data"
    # python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --pos_dims 8 16 24 --h_dims 64 128 --num_layers 2 --dropouts 0 .2 --model_name GCN --pos_method lap-flk --data_name "$data"
    # python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --pos_dims 1 --h_dims 64 128 --num_layers 2 --dropouts 0 .2 --model_name GCN --pos_method lap-full --data_name "$data"
    # python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --pos_dims 8 16 24 --h_dims 64 128 --num_layers 2 --dropouts 0 .2 --model_name GCN --pos_method rwse --data_name "$data"

    # python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --pos_dims 8 16 24 --h_dims 64 128 --num_layers 2 --dropouts 0 .2 --model_name SAGE --pos_method lap-fkf --data_name "$data"
    # python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --pos_dims 8 16 24 --h_dims 64 128 --num_layers 2 --dropouts 0 .2 --model_name SAGE --pos_method spec-attn --data_name "$data"
    # python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --pos_dims 8 16 24 --h_dims 64 128 --num_layers 2 --dropouts 0 .2 --model_name SAGE --pos_method signnet --data_name "$data"
    # python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --pos_dims 1 --h_dims 64 128 --num_layers 2 --dropouts 0 .2 --model_name SAGE --pos_method none --data_name "$data"
    # python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --pos_dims 1 --h_dims 64 128 --num_layers 2 --dropouts 0 .2 --model_name SAGE --pos_method elastic --data_name "$data"
    # python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --pos_dims 8 16 24 --h_dims 64 128 --num_layers 2 --dropouts 0 .2 --model_name SAGE --pos_method lap-fk --data_name "$data"
    # python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --pos_dims 8 16 24 --h_dims 64 128 --num_layers 2 --dropouts 0 .2 --model_name SAGE --pos_method lap-flk --data_name "$data"
    # python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --pos_dims 1 --h_dims 64 128 --num_layers 2 --dropouts 0 .2 --model_name SAGE --pos_method lap-full --data_name "$data"
    # python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --pos_dims 8 16 24 --h_dims 64 128 --num_layers 2 --dropouts 0 .2 --model_name SAGE --pos_method rwse --data_name "$data"
    # python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --h_dims 64 128  --pos_dims 64 128 --num_layers 2 --dropouts 0 .2 --Ks 64 128 --l12s .01 .001 .0001 0 --model_name SAGE_LLPE --pos_method llpe --data_name "$data"

    # python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --pos_dims 8 16 24 --h_dims 64 --num_layers 2 --dropouts 0 .2 --model_name FAGCN --pos_method lap-fkf --data_name "$data"
    python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 --h_dims 128  --pos_dims 64 128 --num_layers 2 --dropouts 0 .2 --Ks 64 128 --l12s .001 .0001 --model_name GCN2_LLPE --pos_method llpe --data_name "$data"
    python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 --pos_dims 8 16 24 --h_dims 128 --num_layers 2 --dropouts 0 .2 --model_name GCN2 --pos_method lap-fkf --data_name "$data"
    python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 --h_dims 128  --pos_dims 64 128 --num_layers 2 --dropouts 0 --Ks 128 --l12s .001 .0001 --model_name FAGCN_LLPE --pos_method llpe --data_name "$data"

    # python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --pos_dims 1 --h_dims 64 128 --num_layers 2 --dropouts 0 --lns 0 .00001 --model_name GT --pos_method none --data_name "$data"
    # python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --pos_dims 1 --h_dims 64 128 --num_layers 2 --dropouts 0 --lns 0 .00001 --model_name GT --pos_method elastic --data_name "$data"
    # python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --pos_dims 8 16 --h_dims 64 128 --num_layers 2 --dropouts 0 --lns 0 .00001 --model_name GT --pos_method lap-fkf --data_name "$data"
    # python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --pos_dims 8 16 --h_dims 64 128 --num_layers 2 --dropouts 0 --lns 0 .00001 --model_name GT --pos_method lap-fk --data_name "$data"
    # python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --pos_dims 8 16 --h_dims 64 128 --num_layers 2 --dropouts 0 --lns 0 .00001 --model_name GT --pos_method lap-flk --data_name "$data"
    # python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --pos_dims 1 --h_dims 64 128 --num_layers 2 --dropouts 0 --lns 0 .00001 --model_name GT --pos_method lap-full --data_name "$data"
    # python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --pos_dims 8 16 --h_dims 64 128 --num_layers 2 --dropouts 0 --lns 0 .00001 --model_name GT --pos_method spec-attn --data_name "$data"
    # python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --pos_dims 8 16 --h_dims 64 128 --num_layers 2 --dropouts 0 --lns 0 .00001 --model_name GT --pos_method signnet --data_name "$data"
    # python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --pos_dims 8 16 --h_dims 64 128 --num_layers 2 --dropouts 0 --lns 0 .00001 --model_name GT --pos_method rwse --data_name "$data"
    # python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --h_dims 64 128  --pos_dims 64 128 --num_layers 2 --dropouts 0 --lns 0 .00001 --Ks 128 --l12s .01 .001 .0001 0 --model_name GT_LLPE --pos_method llpe --data_name "$data"
    
done

# python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --pos_dims 1 --h_dims 64 128 --num_layers 2 --dropouts 0 --lns 0 .00001 --model_name GT --pos_method none --data_name Computers
# python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --pos_dims 1 --h_dims 64 128 --num_layers 2 --dropouts 0 --lns 0 .00001 --model_name GT --pos_method elastic --data_name Computers
# python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --pos_dims 8 16 --h_dims 64 128 --num_layers 2 --dropouts 0 --lns 0 .00001 --model_name GT --pos_method lap-fk --data_name Computers
# python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --pos_dims 8 16 --h_dims 64 128 --num_layers 2 --dropouts 0 --lns 0 .00001 --model_name GT --pos_method lap-flk --data_name Computers
# python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --pos_dims 1 --h_dims 64 128 --num_layers 2 --dropouts 0 --lns 0 .00001 --model_name GT --pos_method lap-full --data_name Computers
# python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --pos_dims 8 16 --h_dims 64 128 --num_layers 2 --dropouts 0 --lns 0 .00001 --model_name GT --pos_method spec-attn --data_name Computers
# python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --pos_dims 8 16 --h_dims 64 128 --num_layers 2 --dropouts 0 --lns 0 .00001 --model_name GT --pos_method signnet --data_name Computers
# python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --pos_dims 8 16 --h_dims 64 128 --num_layers 2 --dropouts 0 --lns 0 .00001 --model_name GT --pos_method rwse --data_name Computers
# python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --h_dims 64 128  --pos_dims 64 128 --num_layers 2 --dropouts 0 --lns 0 .00001 --Ks 128 --l12s .001 .0001 --model_name GT_LLPE --pos_method llpe --data_name Computers

# python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --h_dims 64 128  --pos_dims 64 128 --num_layers 2 --dropouts 0 .2 --Ks 64 128 --l12s .001 .0001 --model_name MLP_LLPE --pos_method llpe --data_name Amazon-ratings
# python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --h_dims 64 128  --pos_dims 64 128 --num_layers 2 --dropouts 0 .2 --Ks 64 128 --l12s .001 .0001 --model_name MLP_LLPE --pos_method llpe --data_name Roman-empire
# python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --h_dims 64 128  --pos_dims 64 128 --num_layers 2 --dropouts 0 .2 --Ks 64 128 --l12s .001 .0001 --model_name SAGE_LLPE --pos_method llpe --data_name Amazon-ratings
# python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --h_dims 64 128  --pos_dims 64 128 --num_layers 2 --dropouts 0 .2 --Ks 64 128 --l12s .001 .0001 --model_name SAGE_LLPE --pos_method llpe --data_name Roman-empire

# python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --h_dims 64 --pos_dims 64 --num_layers 1 --dropouts 0 --lns 0 .00001 --Ks 64 128 --l12s .01 .001 .0001 0 --model_name GT_LLPE --pos_method llpe --data_name Amazon-ratings
# python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --pos_dims 1 --h_dims 64 --num_layers 1 --dropouts 0 --lns 0 .00001 --model_name GT --pos_method none --data_name Amazon-ratings
# python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --pos_dims 1 --h_dims 32 --num_layers 1 --dropouts 0 --lns 0 .00001 --model_name GT --pos_method elastic --data_name Amazon-ratings
# python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --pos_dims 8 16 --h_dims 32 --num_layers 1 --dropouts 0 --lns 0 .00001 --model_name GT --pos_method lap-fk --data_name Amazon-ratings
# python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --pos_dims 8 16 --h_dims 32 --num_layers 1 --dropouts 0 --lns 0 .00001 --model_name GT --pos_method lap-fkf --data_name Amazon-ratings
# python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --pos_dims 8 --h_dims 32 --num_layers 1 --dropouts 0 --lns 0 .00001 --model_name GT --pos_method lap-flk --data_name Amazon-ratings
# python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --pos_dims 1 --h_dims 32 --num_layers 1 --dropouts 0 --lns 0 .00001 --model_name GT --pos_method lap-full --data_name Amazon-ratings
# python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --pos_dims 8 --h_dims 32 --num_layers 1 --dropouts 0 --lns 0 .00001 --model_name GT --pos_method spec-attn --data_name Amazon-ratings
# python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --pos_dims 8 --h_dims 32 --num_layers 1 --dropouts 0 --lns 0 .00001 --model_name GT --pos_method signnet --data_name Amazon-ratings
# python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --pos_dims 8 --h_dims 32 --num_layers 1 --dropouts 0 --lns 0 .00001 --model_name GT --pos_method rwse --data_name Amazon-ratings

# python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --pos_dims 1 --h_dims 64 --num_layers 1 --dropouts 0 --lns 0 .00001 --model_name GT --pos_method none --data_name Roman-empire
# python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --pos_dims 1 --h_dims 64 --num_layers 1 --dropouts 0 --lns 0 .00001 --model_name GT --pos_method elastic --data_name Roman-empire
# python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --pos_dims 8 16 --h_dims 64 --num_layers 1 --dropouts 0 --lns 0 .00001 --model_name GT --pos_method lap-fk --data_name Roman-empire
# python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --pos_dims 8 16 --h_dims 64 --num_layers 1 --dropouts 0 --lns 0 .00001 --model_name GT --pos_method lap-fkf --data_name Roman-empire
# python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --pos_dims 8 16 --h_dims 64 --num_layers 1 --dropouts 0 --lns 0 .00001 --model_name GT --pos_method lap-flk --data_name Roman-empire
# python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --pos_dims 1 --h_dims 64 --num_layers 1 --dropouts 0 --lns 0 .00001 --model_name GT --pos_method lap-full --data_name Roman-empire
# python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --pos_dims 8 16 --h_dims 64 --num_layers 1 --dropouts 0 --lns 0 .00001 --model_name GT --pos_method spec-attn --data_name Roman-empire
# python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --pos_dims 8 16 --h_dims 64 --num_layers 1 --dropouts 0 --lns 0 .00001 --model_name GT --pos_method signnet --data_name Roman-empire
# python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --pos_dims 8 16 --h_dims 64 --num_layers 1 --dropouts 0 --lns 0 .00001 --model_name GT --pos_method rwse --data_name Roman-empire
# python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 5 --h_dims 64  --pos_dims 64 --num_layers 1 --dropouts 0 --lns 0 .00001 --Ks 64 128 --l12s .01 .001 .0001 0 --model_name GT_LLPE --pos_method llpe --data_name Roman-empire




