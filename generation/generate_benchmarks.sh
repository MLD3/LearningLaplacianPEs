#!/bin/sh

# activate correct environment
# conda activate env --- assumes env has the required packages!

# python3 main_generate_benchmarks.py --data_name Cora --data_directory /data/mbito/benchmarks/Cora --save_directory /data/mbito/benchmarks_proc
# python3 main_generate_benchmarks.py --data_name Citeseer --data_directory /data/mbito/benchmarks/Citeseer --save_directory /data/mbito/benchmarks_proc

# python3 main_generate_benchmarks.py --data_name Cora_ML --data_directory /data/mbito/benchmarks/Cora_ML --save_directory /data/mbito/benchmarks_proc
# python3 main_generate_benchmarks.py --data_name Citeseer_full --data_directory /data/mbito/benchmarks/Citeseer_full --save_directory /data/mbito/benchmarks_proc

# python3 main_generate_benchmarks.py --data_name Cornell --data_directory /data/mbito/benchmarks/Cornell --save_directory /data/mbito/benchmarks_proc
# python3 main_generate_benchmarks.py --data_name Wisconsin --data_directory /data/mbito/benchmarks/Wisconsin --save_directory /data/mbito/benchmarks_proc
# python3 main_generate_benchmarks.py --data_name Texas --data_directory /data/mbito/benchmarks/Texas --save_directory /data/mbito/benchmarks_proc

# python3 main_generate_benchmarks.py --data_name Actor --data_directory /data/mbito/benchmarks/Actor --save_directory /data/mbito/benchmarks_proc
# python3 main_generate_benchmarks.py --data_name Squirrel --data_directory /data/mbito/benchmarks/squirrel_filtered --save_directory /data/mbito/benchmarks_proc
# python3 main_generate_benchmarks.py --data_name Chameleon --data_directory /data/mbito/benchmarks/chameleon_filtered --save_directory /data/mbito/benchmarks_proc
# python3 main_generate_benchmarks.py --data_name Squirrel --data_directory /data/mbito/benchmarks/Wikipedia --save_directory /data/mbito/benchmarks_proc
# python3 main_generate_benchmarks.py --data_name Chameleon --data_directory /data/mbito/benchmarks/Wikipedia --save_directory /data/mbito/benchmarks_proc

# python3 main_generate_benchmarks.py --data_name Computers --data_directory /data/mbito/benchmarks/Computers --save_directory /data/mbito/benchmarks_proc
# python3 main_generate_benchmarks.py --data_name Photo --data_directory /data/mbito/benchmarks/Photo --save_directory /data/mbito/benchmarks_proc

# python3 main_generate_benchmarks.py --data_name Amazon-ratings --data_directory /data/mbito/benchmarks/Amazon-ratings --save_directory /data/mbito/benchmarks_proc
# python3 main_generate_benchmarks.py --data_name Roman-empire --data_directory /data/mbito/benchmarks/Roman-empire --save_directory /data/mbito/benchmarks_proc
# python3 main_generate_benchmarks.py --data_name Minesweeper --data_directory /data/mbito/benchmarks/Minesweeper --save_directory /data/mbito/benchmarks_proc
# python3 main_generate_benchmarks.py --data_name Tolokers --data_directory /data/mbito/benchmarks/Tolokers --save_directory /data/mbito/benchmarks_proc

python3 main_generate_benchmarks.py --data_name Penn94 --data_directory /data/mbito/benchmarks/Penn94 --save_directory /data/mbito/benchmarks_proc
python3 main_generate_benchmarks.py --data_name Questions --data_directory /data/mbito/benchmarks/Questions --save_directory /data/mbito/benchmarks_proc



