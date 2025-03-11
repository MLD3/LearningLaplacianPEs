# Learning Laplacian Positional Encodings for Heterophilous Graphs
[AISTATS 2025] Learning Laplacian Positional Encodings for Heterophilous Graphs

## Instructions for Reproducibility
1. In the `generation` directory, use the command `source generate_benchmarks.sh` to download and preprocess datasets from PyTorch Geometric.
2. Once the datasets are downloaded and preprocessed, use the command `source search.sh` to train models and save results to the `results` directory.
3. After training the models and saving the results, load the results in the `load_results.ipynb` notebook located in the `results` directory.

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{ItoKW25llpe,
  author       = {Michael Ito and Jiong Zhu and Dexiong Chen and Danai Koutra and Jenna Wiens},
  title        = {Learning Laplacian Positional Encodings for Heterophilous Graphs},
  booktitle    = {International Conference on Artificial Intelligence and Statistics},
  publisher    = {PMLR},
  year         = {2025},
}
