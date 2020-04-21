# MacroNASBenchmark
These NAS benchmarks for CIFAR10/100 are used in the paper:
> Den Ottelander T., Dushatskiy A., Virgolin M., and Bosman P. A. N.: Local Search is a Remarkably Strong Baseline for Neural Architecture Search. arXiv:2004.08996 (2020) https://arxiv.org/abs/2004.08996

## Features
These NAS benchmarks aim at providing the community with a new tool for evaluating the performance of search algorithms for NAS. Some features which make it different from existing alternative approaches:
* Search is performed at macro-level: cell types are the discrete variables, they are not repeated, the number of cells is flexible
* All solutions in the considered search space are feasible to make the usage easier with almost any search algorithm
* We believe it's better suited for benchmarking multi-objective optimization algorithms (see the paper)
* Quite a large search space: **>200K** unique architectures

## Benchmarks description
* All considered architectures consist of **14** searchable and some fixed cells
* Each cell can have three options: 
  1. Identity: a placeholder to allow nets of different depths (encoded as '**I**')
  2. MBConv with expansion rate 3, kernel size 3x3 (encoded as '**1**')
  3. MBConv with expansion rate 6, kernel size 5x5 (encoded as '**2**')
* A net from the search space having the best validation accuracy on CIFAR10 (pic by [@tdenottelander](https://github.com/tdenottelander)):
<img src="https://github.com/ArkadiyD/MacroNASBenchmark/blob/master/benchmarks_statistics_plots/net_example.png"  width="800"/> 

## Benchmarks usage
To test your search algorithm on benchmarks:
* ```git clone```
* Data for CIFAR/CIFAR100 stored in `benchmark_cifar10_dataset.json`, `benchmark_cifar100_dataset.json` respectively
* In json files there are all (possibly including identity cells) models of the search space (>4M) as keys, solutions encoding is described above

## Benchmarks creation & reproducibility
To reproduce benchmarks or change some hyperparameters and obtain your own version of them (an example for CIFAR10):

0. ```git clone```
1. ```pip3 install --user -r requirements.txt```
2. Trained supernet models are provided in `benchmark_cifar10/`. If you want to re-train, delete existing models and run: `./scripts/train_supernet_cifar10.sh`
   * By default, GPU #0 is used
   * When using for the first time, CIFAR dataset will be downloaded and placed to `/datasets`
   * Data split is fixed and stored in `benchmark_cifar10/`
   * The trained supernet (with and without SWA ensembling) will be placed in `benchmark_cifar10/`
   
3. To evaluate models: `./scripts/do_evaluations_cifar10.sh`
   * By default, GPU #0 is used. 
   * Parameters `first_net_id, last_net_id` determine which nets will be evaluated (only first 1000 nets by default). 
   * All evaluated solutions data will be put in `benchmark_cifar10/evaluations/`. Data for each model is stored in a separate `.json`
   * You can run multiple copies of this script to do evaluations in parallel. 

4. To build the final dataset:`python3 src/build_dataset.py --dir benchmark_cifar10`
   * Will assemble separate `json`s into a single `json` (e.g., `benchmark_cifar10_dataset.json`) containing the data for all evaluated solutions

## Benchmarks statistics
<img src="https://github.com/ArkadiyD/MacroNASBenchmark/blob/master/benchmarks_statistics_plots/CIFAR10_mmacs_val.png" width="300"/> <img src="https://github.com/ArkadiyD/MacroNASBenchmark/blob/master/benchmarks_statistics_plots/CIFAR10_mmacs_test.png" width="300"/>

<img src="https://github.com/ArkadiyD/MacroNASBenchmark/blob/master/benchmarks_statistics_plots/CIFAR100_mmacs_val.png" width="300"/> <img src="https://github.com/ArkadiyD/MacroNASBenchmark/blob/master/benchmarks_statistics_plots/CIFAR100_mmacs_test.png" width="300"/>

<img src="https://github.com/ArkadiyD/MacroNASBenchmark/blob/master/benchmarks_statistics_plots/CIFAR10_val_test.png" width="300"/> <img src="https://github.com/ArkadiyD/MacroNASBenchmark/blob/master/benchmarks_statistics_plots/CIFAR100_val_test.png" width="300"/>
