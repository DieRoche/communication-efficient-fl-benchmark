# Communication-Efficient Federated Learning Benchmark

> **Attribution notice:** This repository does not claim authorship of the original FLoCoRA, FedQClip, or SparsyFed methods or their original implementations. The corresponding method directories contain adapted or modified versions of the external repositories identified below; the original authors retain authorship of their methods and original code. This repository contributes standardized configurations, experimental adaptations, implementation corrections where identifiable, communication- and computational-cost measurement, logging, reporting, and comparative evaluation for the accompanying research. The root FedAvg code is a separate PyTorch implementation/adaptation of Federated Averaging, not a direct copy, translation, or port of the cited TensorFlow/TensorFlow Federated reference.

## Repository Overview

This repository is a method-independent reproducibility and standardized-evaluation workspace for communication-efficient federated learning (FL). The benchmark instrumentation and experimental workflow can be used with other FL implementations; using FedQClip, FLoCoRA, or SparsyFed is not required. Their local adaptations provide concrete reference cases for quantization, low-rank adaptation, and sparsification, respectively, and demonstrate how those technique families can be measured under the benchmark.

> **Snapshot status:** This checkout is not self-contained. It has no dependency/environment manifest or root license, and several entry points import files or packages absent from this tree. The commands below are the entry points implied by the checked-in code, but they cannot all be verified end to end until the missing files and environments are restored. No benchmark result files are committed here.

## Standardized Configurations

Configuration is deliberately kept close to each implementation:

| Component | Configuration | Verified defaults in this snapshot |
|---|---|---|
| Root PyTorch FedAvg | [`config.py`](config.py) | CIFAR-10, ResNet-18 (`resnet` selector), 100 total clients, participation fraction 0.1 (10 active clients), 100 rounds, 5 local epochs, batch size 128, learning rate 0.01, Dirichlet alpha 0.5, train fraction 0.8, seed 5; optional CSR sparse masking and 8/16-bit uplink quantization |
| FedQClip | [`fedqclip/config.py`](fedqclip/config.py) | CIFAR-10, ResNet selector, 100 total clients, participation fraction 0.1 (10 active clients), 100 rounds, 5 local epochs, batch size 128, learning rate 0.01, Dirichlet alpha 0.5, seed 5, 8-bit quantization enabled |
| FLoCoRA | [`flocora/args.py`](flocora/args.py) | CIFAR-10, ResNet-18, 100 clients, sampling rate 0.1, 100 rounds, 5 local epochs, batch size 128, learning rate 0.01, LDA alpha 0.5, validation ratio 0.2, seed 5, LoRA rank/alpha 16 |
| SparsyFed (Hydra root) | [`sparsyfed/project/conf/cifar_resnet18.yaml`](sparsyfed/project/conf/cifar_resnet18.yaml) | Composes the CIFAR task, federated, strategy, and dataset groups; selects `fedavgNZ`; W&B is disabled by default |
| SparsyFed dataset | [`cifar_lda.yaml`](sparsyfed/project/conf/dataset/cifar_lda.yaml) | 100 clients, 10 classes, non-IID LDA partitioning with alpha 0.5, validation ratio 0.2, seed 5 |
| SparsyFed federation | [`cifar_resnet18.yaml`](sparsyfed/project/conf/fed/cifar_resnet18.yaml) | 100 total clients, 10 training and 10 evaluation clients per round, 100 rounds, client-selection seed 5 |
| SparsyFed task | [`cifar_resnet18.yaml`](sparsyfed/project/conf/task/cifar_resnet18.yaml) | SparsyFed ResNet-18, sparsity 0.90, 5 local epochs, batch size 128, learning rate 0.01 |
| SparsyFed strategies | [`conf/strategy/`](sparsyfed/project/conf/strategy/) | Hydra targets for FedAvg, FedAvgNZ, adaptive server optimizers, and other included strategies |

The root, FedQClip, FLoCoRA, and default SparsyFed CIFAR configurations align on CIFAR-10, 100 total clients, 10 active clients per round (a 0.1 participation/sample rate), Dirichlet/LDA alpha 0.5, five local epochs, batch size 128, learning rate 0.01, and seed 5. Root, FedQClip, and FLoCoRA expose these settings as command-line options; SparsyFed uses Hydra overrides.

SparsyFed also includes complete configuration groups for [CUB-200 with ViT](sparsyfed/project/conf/cub_vit.yaml) and [Speech Commands with ResNet-18](sparsyfed/project/conf/speech_resnet18.yaml). These are additional task configurations, not evidence that every method has been aligned or evaluated on those tasks.

## Original Implementations and Code Provenance

| Local component | Method | Original or reference source | Relationship to this repository | License or attribution status |
|---|---|---|---|---|
| [`fedqclip/`](fedqclip/) | FedQClip | [jianinghui/FedQClip](https://github.com/jianinghui/FedQClip) | Adapted or modified from the source implementation as a quantization reference case for standardized evaluation and added reporting/instrumentation. It is not required in order to use the benchmark. | No local license or notice is included; consult the upstream repository and authors. |
| [`flocora/`](flocora/) | FLoCoRA | [lgrativol/flocora_eusipco24](https://github.com/lgrativol/flocora_eusipco24) | Adapted subset of the repository associated with the EUSIPCO 2024 work and used as a low-rank reference case. The upstream code is built on Flower; this implementation is not required in order to use the benchmark. | No explicit license was identified; users should consult the original repository and authors regarding reuse conditions. |
| [`sparsyfed/`](sparsyfed/) | SparsyFed | [AGuastella/sparsyfed](https://github.com/AGuastella/sparsyfed) | Adapted version used as a sparsification reference case under standardized evaluation conditions; it is not required in order to use the benchmark. The upstream repository identifies itself as the official implementation of “SparsyFed: Sparse Adaptive Federated Training.” | Upstream is Apache License 2.0, but its `LICENSE` file is absent from this snapshot; restore/preserve it before redistribution. |
| [`main.py`](main.py), [`config.py`](config.py), [`data_utils.py`](data_utils.py), [`ResNet18.py`](ResNet18.py) | FedAvg | McMahan et al. (AISTATS 2017); [Google DP-FTRL FedAvg reference](https://github.com/google-research/federated/blob/master/dp_ftrl/dp_fedavg.py) | Local PyTorch implementation/adaptation of the Federated Averaging training process, with optional sparse/quantized payload instrumentation. | Not a direct copy, translation, or port of the Google file. No root license is included. |

The Google reference uses TensorFlow and TensorFlow Federated and contains differential-privacy-related adaptations. It is listed only as a conceptual/structural reference for the FedAvg training process; it is not described here as the original FedAvg implementation.

## Contributions of This Repository

### Standardized experimental configuration

- Exposes common controls for client counts and participation, rounds, local epochs, batch size, learning rate, seeds, CIFAR data selection, and Dirichlet/LDA partitioning.
- Provides a matched CIFAR-10/ResNet-18 configuration path for the included reference cases, while making remaining default differences visible. The same evaluation approach can be applied to other FL implementations.
- Adds Hydra configuration composition for SparsyFed datasets, tasks, federated execution, and strategies.

### Evaluation instrumentation

- Root FedAvg reports accuracy, loss, client statistics, per-round/cumulative upload and download bytes, sparse density, and estimated compression/decompression and masking operations to Weights & Biases.
- FedQClip reports validation accuracy/loss, active clients, per-client and aggregate traffic, sparsity, training/evaluation estimates, and quantization/dequantization estimates.
- FLoCoRA aggregates client training, evaluation, aggregation, serialization, compression, and decompression estimates and writes experiment histories under its configured results path.
- SparsyFed includes traffic accounting in [`traffic.py`](sparsyfed/project/fed/utils/traffic.py), W&B history/server integration, task losses/accuracies, sparsity metrics, and file-based histories/checkpoints.
- [`wandb_tests.py`](wandb_tests.py) exports selected W&B histories to Excel, and [`ploter.py`](ploter.py) creates accuracy-versus-traffic/FLOP figures from Excel workbooks.

These operation counts are code-level estimates, not hardware profiler measurements; interpret them according to each implementation's estimator.

### Functional adaptations

- Root FedAvg adds magnitude-based sparse masking, CSR payload serialization/reconstruction, optional 8/16-bit transport quantization, and active-client-only traffic accounting.
- FedQClip contains clipped client/server quantization and explicit byte-packet accounting around model updates.
- FLoCoRA exposes LoRA/LoHa strategy selection, optional fake quantization, Flower/Ray resource selection, and standardized report metadata.
- SparsyFed composes sparse task modules, deterministic client selection, custom Flower strategies, and configurable result/checkpoint handling.

### Corrections

Comments and guards in the checked-in code identify defensive handling for zero-valued quantization scales, missing child-process results, non-positive Ray CPU allocation, bounded Ray GPU allocation, and active-client traffic counts. A complete bug-fix history cannot be established because upstream commit identifiers and a patch history are not included.

## Evaluated Methods and Reference Entry Points

- **FedAvg:** weighted aggregation and the benchmark's root experiment/instrumentation are implemented in [`main.py`](main.py), configured by [`config.py`](config.py).
- **Quantization reference — FedQClip:** quantized, clipped federated SGD/update handling is provided in [`fedqclip/FedQClip.py`](fedqclip/FedQClip.py).
- **Low-rank reference — FLoCoRA:** low-rank federated adaptation is driven by the Flower simulation entry point [`flocora/main_ray.py`](flocora/main_ray.py).
- **Sparsification reference — SparsyFed:** sparse adaptive training is launched through [`sparsyfed/project/main.py`](sparsyfed/project/main.py), with task-specific sparse modules and custom aggregation strategies.

These are included reference implementations and evaluation targets, not mandatory dependencies of the benchmark design. Another FL method can use the same standardized configuration, logging, communication-cost, computational-cost, reporting, and comparison conventions after integrating the corresponding instrumentation. The presence of code or configuration denotes benchmark support; it does not by itself establish a completed result for every combination.

## Repository Structure

```text
.
├── README.md                  # Provenance, configuration, and reproducibility guide
├── config.py                  # Root FedAvg CLI configuration
├── main.py                    # Root PyTorch FedAvg experiment and instrumentation
├── data_utils.py              # CIFAR/MNIST loading and client partitioning
├── ResNet18.py                # Root model definition
├── fedqclip/
│   ├── config.py              # FedQClip CLI configuration
│   └── FedQClip.py            # Adapted training, quantization, and reporting
├── flocora/
│   ├── args.py                # FLoCoRA CLI configuration
│   ├── main_ray.py            # Flower/Ray simulation entry point
│   ├── client.py              # Flower client wrapper
│   └── log.py                 # Console/file logging
├── sparsyfed/project/
│   ├── main.py                # Hydra/Flower simulation entry point
│   ├── conf/                  # Dataset, task, federation, and strategy YAML
│   ├── client/                # Flower client construction
│   ├── fed/                   # Servers, strategies, traffic, masks, and plots
│   ├── task/                  # CIFAR, CUB, Speech, and sparse model/task code
│   ├── dispatch/              # Runtime configuration dispatch
│   ├── types/                 # Shared types
│   └── utils/                 # Runtime and diagnostics utilities
├── wandb_tests.py             # W&B-to-Excel metric export
└── ploter.py                  # Excel-based comparative PDF plotting
```

### SparsyFed configuration structure

[`sparsyfed/project/conf/`](sparsyfed/project/conf/) is a Hydra configuration hierarchy. A top-level experiment file selects one item from each configuration group through its `defaults` list; command-line Hydra overrides can replace any composed value.

```text
sparsyfed/project/conf/
├── cifar_resnet18.yaml        # Default CIFAR experiment composition and output/W&B controls
├── cub_vit.yaml               # Default CUB-200/ViT experiment composition
├── speech_resnet18.yaml       # Default Speech Commands/ResNet-18 composition
├── dataset/
│   ├── cifar_lda.yaml         # Paths, client count, classes, split, seed, and LDA/IID controls
│   ├── cub_200_2011.yaml      # CUB paths and client-partition parameters
│   └── speech_lda.yaml        # Speech Commands paths and client-partition parameters
├── fed/
│   ├── cifar_resnet18.yaml    # Rounds, participating/evaluation clients, resources, and checkpoints
│   ├── cub_vit.yaml           # CUB federated execution and checkpoint settings
│   └── speech_resnet18.yaml   # Speech federated execution and checkpoint settings
├── task/
│   ├── cifar_resnet18.yaml    # Model/train dispatch, sparsity, batches, epochs, LR, and metrics
│   ├── cub_vit.yaml           # ViT task selection, sparsity, optimizer/training, and metrics
│   └── speech_resnet18.yaml   # Speech model/train selection, sparsity, training, and metrics
└── strategy/
    ├── fedavg.yaml            # Flower FedAvg target
    ├── fedavgNZ.yaml          # Local nonzero-aware aggregation target used by default for CIFAR
    ├── fedavgDynamics.yaml    # Local dynamic sparse strategy target
    ├── fedavgFLASH.yaml       # Local FLASH strategy with task sparsity interpolation
    ├── fedavgHFLASH.yaml      # Local heterogeneous FLASH target
    ├── fedavgHetero.yaml      # Local heterogeneous aggregation target
    ├── fedadam_custom.yaml    # Local FedAdam implementation target
    └── fedadagrad/fedadam/fedavgm/fedyogi.yaml
                               # Flower adaptive and momentum strategy targets
```

The top-level files also control output-directory reuse, temporary working files, checkpoint-saving frequency, cleanup patterns, local client tests, Ray settings, and W&B setup. The `dataset/` group controls how data is located and partitioned; `fed/` controls the federation and runtime resources; `task/` binds model/data dispatch to local training and evaluation; and `strategy/` selects the Flower or local server aggregation class. These groups are independently overridable, which is what allows a common task to be evaluated with different strategies without duplicating the entire experiment configuration.

## Environment and Dependencies

No `requirements.txt`, `environment.yml`, `pyproject.toml`, lockfile, or container definition exists in this checkout. Imports show that the components require overlapping but non-identical Python stacks including PyTorch/torchvision, NumPy, Matplotlib, pandas/openpyxl, Weights & Biases, Flower, Ray, Hydra/OmegaConf, SciPy, scikit-learn, and task-specific packages.

Do not assume one environment works for every component.

- **FedAvg:** TODO: add and verify a root environment manifest.
- **FedQClip:** TODO: restore the method environment/requirements and missing imported modules.
- **FLoCoRA:** TODO: restore its `requirements.txt` and omitted `utils/`, `strategies/`, and model files from the adapted source, then pin the tested versions.
- **SparsyFed:** TODO: restore its upstream `pyproject.toml`, lockfile, setup script, and license, or add an equivalent verified environment manifest.

## Dataset Preparation

Root FedAvg and FedQClip call [`data_utils.get_dataset`](data_utils.py), which downloads CIFAR-10/CIFAR-100 or MNIST through torchvision, splits the training set, and creates client partitions using IID splitting when `dirichlet == 0` or a Dirichlet label distribution otherwise. Data is placed under `./data` relative to the launch directory. The helper also writes `client_distributions.png` and `client_0_inspection.png`.

FLoCoRA expects `--dataset_path` (default `./data`) and delegates download/partition creation to omitted `utils.dataset` functions; dataset preparation cannot be verified from this snapshot.

For SparsyFed CIFAR, configure [`conf/dataset/cifar_lda.yaml`](sparsyfed/project/conf/dataset/cifar_lda.yaml), then—after restoring its environment—run from `sparsyfed/`:

```bash
python -m project.task.cifar_resnet18.dataset_preparation
```

Equivalent checked-in preparation entry points exist for CUB and Speech Commands under their task directories.

## Running the Experiments

The following commands reflect the checked-in entry points. They are **not currently end-to-end verified** because the snapshot lacks dependencies and/or imported files.

### FedAvg

From the repository root:

```bash
python main.py --n_client 100 --client_fraction 0.1 --n_epoch 100
```

Primary configuration: [`config.py`](config.py). W&B is disabled by default. To opt in, pass `--wandb_enabled true` and optionally `--wandb_project <project>`; the neutral default project is `communication-efficient-fl-benchmark`, under the user's currently authenticated W&B account. Dataset diagnostic PNGs are local. The command currently fails before execution because `compression.py` and lowercase `resnet18.py` are absent.

### FedQClip

From `fedqclip/`:

```bash
python FedQClip.py
```

Primary configuration: [`config.py`](fedqclip/config.py). W&B is disabled by default; opt in with `--wandb_enabled true` and optionally `--wandb_project <project>`. The script writes `trainloss_<model>.txt` in its working directory. It currently depends on absent `data_utils.py`, `ResNet18.py`, and `effnet.py` in the method's import context; its exact launch/PYTHONPATH arrangement must be restored and verified.

### FLoCoRA

From `flocora/`:

```bash
python main_ray.py --strategy fedlora --dataset cifar10 --model resnet18
```

Primary configuration: [`args.py`](flocora/args.py). Results are intended for `--path_results` (default `results/`), and `log.py` writes `log.log`. W&B is disabled by default; opt in with `--wandb`, optionally adding `--entity <entity>` and `--wandb_prj_name <project>`. No entity is embedded in the repository. The command cannot run until the omitted upstream/adapted modules are restored.

### SparsyFed

From `sparsyfed/`, after restoring a compatible environment:

```bash
python -m project.main --config-name=cifar_resnet18
```

Hydra accepts overrides such as `use_wandb=true`, `wandb.setup.project=<project>`, `wandb.setup.entity=<entity>`, `task.sparsity=0.95`, or `fed.num_clients_per_round=10`. W&B is disabled by default and its entity defaults to `null`. Hydra creates an output directory; the program creates `results/` and `working/` beneath it (or uses `reuse_output_dir`/`working_dir`) and saves configured history, YAML, log, and parameter artifacts. Dependency installation remains TODO because no local manifest is included.

## Communication-Cost Measurement

Traffic values are represented as bytes in the root and FedQClip implementations. Root FedAvg measures serialized client payload sizes (dense or CSR, with quantized value widths where enabled), counts a full dense global model download for each active client, and reports per-round upload/download plus cumulative overall traffic. FedQClip serializes quantized packets, measures their lengths, and likewise multiplies the server packet by active participants. FLoCoRA derives message sizes from model/quantized parameters and passes model size, clients per round, and round count into history reporting. SparsyFed's [`traffic.py`](sparsyfed/project/fed/utils/traffic.py) provides dense and sparse object-size accounting used by its server strategies.

No network-protocol, transport, encryption, or framework framing overhead beyond the explicitly serialized objects is claimed.

## Computational-Cost Measurement

Implemented computational metrics include:

- Root FedAvg: estimated sparse-mask selection, CSR/quantization compression, and reconstruction/dequantization operations, per round and cumulatively.
- FedQClip: estimated client training, server evaluation, compression, and decompression operations.
- FLoCoRA: aggregated training, evaluation, server aggregation, serialization, client/server compression, and client/server decompression estimates where supplied by its omitted utility modules.
- SparsyFed: training/evaluation metrics and sparse model diagnostics; this snapshot does not support a blanket claim that all root/FLoCoRA FLOP fields are implemented in its execution path.

The repository does not include measured runtime, energy, latency, or hardware-counter results.

## Generated Metrics and Outputs

Depending on the method and enabled logging, implemented fields include test/validation accuracy, training and evaluation loss, active-client counts, sparsity/density, uplink/download/overall traffic, per-client traffic, round and cumulative operation estimates, and compression/decompression totals. Root and FedQClip primarily log to W&B; FLoCoRA writes histories under `path_results`; SparsyFed saves Hydra output artifacts and optionally logs to W&B.

Generated data, model checkpoints, W&B exports, spreadsheets, and PDFs are not committed in this repository snapshot.

## Analysis, Tables, and Figures

[`wandb_tests.py`](wandb_tests.py) queries W&B runs, extracts configured metrics, and exports per-run Excel workbooks. It refuses to run unless `WANDB_ENTITY` and `WANDB_GROUP` are explicitly set; `WANDB_PROJECT` is optional and defaults to `communication-efficient-fl-benchmark`. This analysis script is the only place that reads existing remote runs, and invoking it is an explicit network operation.

[`ploter.py`](ploter.py) reads Excel sheets such as accuracy, overall traffic, round FLOPs, and client/server compression/decompression estimates, then saves comparative PDF figures. Its input folder is an absolute developer-local Windows path, so update `EXCEL_FILES_FOLDER` before running:

```bash
python ploter.py
```

## Differences from the Upstream Implementations

### FedQClip

The local single-file experiment is reconfigured for the benchmark's CIFAR/ResNet settings and instrumented with explicit packet sizing, active-client uplink/downlink reporting, sparsity summaries, and estimated training/evaluation and quantization costs. **The exact upstream commit and full change history should be documented before publication.**

### FLoCoRA

The local subset adapts Flower simulation arguments and adds standardized W&B names/metadata, client metric aggregation, compute/traffic report inputs, safer child-process error reporting, and bounded Ray resource selection. Many upstream files are not included. **The exact upstream commit and full change history should be documented before publication.**

### SparsyFed

The local subset retains the Hydra/Flower task architecture while reconfiguring the default CIFAR experiment for 100 total clients, 10 active clients per round, 100 rounds, seed 5, and optional W&B comparative logging. It includes traffic and custom strategy adaptations but omits upstream packaging, setup, plotting, and license files. **The exact upstream commit and full change history should be documented before publication.**

### FedAvg

The root implementation uses PyTorch and torchvision, implements weighted client-delta aggregation, and adds optional magnitude masking, CSR encoding, quantization, communication accounting, and estimated compression costs. It is based on the FedAvg algorithm, not on copied TensorFlow/TensorFlow Federated code. **The exact development history should be documented before publication.**

## Reproducibility Notes

- Record the commit of this repository and the precise upstream commit used for each adapted directory; those identifiers are currently absent.
- Restore and pin a separate verified environment for each implementation.
- Align client totals and active clients explicitly rather than relying on the differing defaults.
- Record all CLI arguments or resolved Hydra YAML, dataset versions/paths, partition artifacts, random seeds, device/software versions, and W&B run identifiers.
- Treat W&B as an optional external service containing run data not present in this repository.
- Verify case-sensitive imports on Linux (`ResNet18.py` versus `resnet18.py`) and restore every imported module before running.
- Preserve generated client partitions when exact cross-method data reuse is required.

## Citation

Accompanying research (citation TODO) and the original method papers:

```bibtex
@inproceedings{McMahan2017CommunicationEfficient,
  title     = {Communication-Efficient Learning of Deep Networks from Decentralized Data},
  booktitle = {International Conference on Artificial Intelligence and Statistics (AISTATS)},
  author    = {McMahan, Brendan and Moore, Eider and Ramage, Daniel and Hampson, Seth and y Arcas, Blaise Aguera},
  year      = {2017},
  volume    = {54},
  pages     = {1273--1282},
  publisher = {PMLR}
}

@article{Qu2025FedQClip,
  title   = {FedQClip: Accelerating Federated Learning via Quantized Clipped SGD},
  author  = {Qu, Zhihao and Jia, Ninghui and Ye, Baoliu and Hu, Shihong and Guo, Song},
  year    = {2025},
  journal = {IEEE Transactions on Computers},
  volume  = {74},
  number  = {2},
  pages   = {717--730},
  doi     = {10.1109/TC.2024.3477972}
}

@inproceedings{Grativol2024FLoCoRA,
  title     = {FLoCoRA: Federated Learning Compression with Low-Rank Adaptation},
  booktitle = {European Signal Processing Conference (EUSIPCO)},
  author    = {Grativol, Lucas Ribeiro and Leonardon, Mathieu and Muller, Guillaume and Fresse, Virginie and Arzel, Matthieu},
  year      = {2024},
  address   = {Lyon, France},
  eprint    = {2406.14082},
  doi       = {10.48550/arXiv.2406.14082}
}

@inproceedings{Guastella2025SparsyFed,
  title     = {SparsyFed: Sparse Adaptive Federated Learning},
  booktitle = {International Conference on Representation Learning (ICLR)},
  author    = {Guastella, Adriano and Sani, Lorenzo and Iacob, Alex and Mora, Alessio and Bellavista, Paolo and Lane, Nic},
  year      = {2025},
  pages     = {14781--14813},
  address   = {Singapore}
}
```


## Licenses and Third-Party Code

No root `LICENSE`, `COPYING`, or `NOTICE` file, and no such file within the three imported method directories, is present in this snapshot. Consequently, this repository does not currently state a license for its original evaluation material.

Any license later selected for this repository applies only to original material created for this evaluation framework unless explicitly stated otherwise. Third-party code remains subject to its original copyright and license conditions. Existing license and notice files from imported projects must be restored and preserved. In particular, the SparsyFed upstream repository declares Apache License 2.0, but that does not automatically license unrelated benchmark code. Consult each upstream repository and its authors before reusing or redistributing third-party components.
