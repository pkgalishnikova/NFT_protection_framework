# StegaStamp: Baseline and Prior Work

This folder contains all work related to StegaStamp (Tancik et al., 2020), which was a preparatory work to understand the pipeline and the primary baseline for comparison with HiDDeN.

## Files

- `Invisible Hyperlinks in Physical Photographs/`: Reproduction of the full StegaStamp model from the original paper. This work was conducted as a preliminary study before adapting HiDDeN for the NFT verification case.
- `StegaStamp_baseline_scripts/`: Scripts with variations of StegaStamp model and evaluations of them on the dataset.
- `StegaStamp_base_model/`: Scripts and configuration files used fas a baseline for tuning the StegaStamp model on the target dataset.
- `StegaStamp_exp_06_tuning/`: Experiment 06 of StegaStamp base model tuning, which was considered the best and chosen for further analysis.
- `StegaStamp_experiments/`: Variations, metrics, descriptions and sample images from all StegaStamp experimental runs.

## Why StegaStamp was not chosen as a final model

The original StegaStamp design was originally created for a task of physical photograph recovery, and its spatial transformer network and complicated upsampling decoder create distortions that are unnecessary for the structured embedding task required in this work. Instead of accumulating fixes on top of an architecture that was not designed for this purpose, it was decided to switch from scratch to a model that better meets the requirements of the task.
