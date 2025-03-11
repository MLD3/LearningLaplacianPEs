# Learning Laplacian Positional Encodings for Heterophilous Graphs
[AISTATS 2025] Learning Laplacian Positional Encodings for Heterophilous Graphs

## Instructions for Reproducibility
1. In the `generation` directory, use the following code to download and preprocess datasets from PyTorch Geometric:
   ```bash
   source generate_benchmarks.sh
   ```
2. Once the datasets are downloaded and preprocessed, use the following code to train models and save results to the `results` directory.
   ```bash
   source search.sh
   ```
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
```
