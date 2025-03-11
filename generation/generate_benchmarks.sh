#!/bin/sh

# activate correct environment
# conda activate env --- assumes env has the required packages

# python3 main_generate_benchmarks.py --data_name Cora --data_directory ../benchmarks/Cora --save_directory ../benchmarks_proc
# python3 main_generate_benchmarks.py --data_name Cora_ML --data_directory ../benchmarks/Cora_ML --save_directory ../benchmarks_proc

# python3 main_generate_benchmarks.py --data_name Cornell --data_directory ../benchmarks/Cornell --save_directory ../benchmarks_proc
# python3 main_generate_benchmarks.py --data_name Wisconsin --data_directory ../benchmarks/Wisconsin --save_directory ../benchmarks_proc
# python3 main_generate_benchmarks.py --data_name Texas --data_directory ../benchmarks/Texas --save_directory ../benchmarks_proc

# python3 main_generate_benchmarks.py --data_name Computers --data_directory ../benchmarks/Computers --save_directory ../benchmarks_proc
# python3 main_generate_benchmarks.py --data_name Photo --data_directory ../benchmarks/Photo --save_directory ../benchmarks_proc

# python3 main_generate_benchmarks.py --data_name Amazon-ratings --data_directory ../benchmarks/Amazon-ratings --save_directory ../benchmarks_proc
# python3 main_generate_benchmarks.py --data_name Roman-empire --data_directory ../benchmarks/Roman-empire --save_directory ../benchmarks_proc
# python3 main_generate_benchmarks.py --data_name Minesweeper --data_directory ../benchmarks/Minesweeper --save_directory ../benchmarks_proc
# python3 main_generate_benchmarks.py --data_name Tolokers --data_directory ../benchmarks/Tolokers --save_directory ../benchmarks_proc

python3 main_generate_benchmarks.py --data_name Penn94 --data_directory ../benchmarks/Penn94 --save_directory ../benchmarks_proc
python3 main_generate_benchmarks.py --data_name Questions --data_directory ..benchmarks/Questions --save_directory ../benchmarks_proc



