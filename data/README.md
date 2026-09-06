 Data

## Source

This project uses the **5G High Density Demand (HDD) Dataset in Liverpool City Region, UK**, specifically the **ACC Arena** scenario.

The original ACC Arena dataset contains measurements for **12,000 users** across **10,000 timestamps** and includes:

- user position
- traffic type
- Radio Unit (RU) association
- downlink and uplink SINR
- throughput
- Physical Resource Block (PRB) utilization
- Block Error Rate (BLER)

The dataset was generated using a digital-twin-based model and a 5G system-level simulator and was experimentally validated against measurements collected in the Liverpool City Region High Density Demand project.

## Why the dataset is not included

The original dataset is publicly available and intended for research, machine learning, network analysis, prediction, experimentation, and prototyping.

However, this repository does **not redistribute either the original dataset or the derived ACC Arena subset**.

The associated Scientific Data publication is licensed under **Creative Commons Attribution 4.0 (CC BY 4.0)**, but this repository does not assume that the same licence automatically applies to the downloadable dataset files themselves.

Instead, the repository versions the **data extraction pipeline** used to reproduce the subset required by the experiments.

## Reproducing the ACC Arena subset

Download the original ACC Arena dataset and run:

```bash
python scripts/extract_acc_arena_subset.py \
    --dataset-root "/path/to/ACC Arena" \
    --output-path data/raw/acc_arena_subset.csv \
    --measurement-device-count 20 \
    --target-user-start 20 \
    --target-user-count 200 \
    --time-stride 20

The generated file will be:

data/raw/acc_arena_subset.csv

This is the dataset consumed by the analysis notebook.

Subset configuration

The experiments in this repository use:

20 operator measurement devices: user IDs 0-19
200 target users: user IDs 20-219
220 users in total
time stride: 20
108,240 rows in the resulting subset

The extraction script converts the original metric-specific CSV shards into a single tidy table containing one row per (user_id, timestamp).

Repository policy

Files under:

data/raw/

are intentionally excluded from Git version control.

The extraction logic is versioned instead, allowing the experimental input to be reproduced from the original dataset without redistributing third-party data.

Dataset citation

Maheshwari, M. K., Raschellà, A., Mackay, M. et al.
5G High Density Demand Dataset in Liverpool City Region, UK.
Scientific Data 12, 1992 (2025).
DOI: 10.1038/s41597-025-06282-0