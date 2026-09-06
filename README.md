# 5G Throughput Prediction in a Dense Deployment with Federated Learning

Machine-learning pipeline for predicting **per-user throughput in a dense 5G deployment** using radio and network measurements.

The project compares centralized **Random Forest** and **Multi-Layer Perceptron (MLP)** regressors, studies the impact of operator measurement devices as network-state features, and simulates **per-user Federated Learning with FedAvg** using PyTorch.

---

## Problem

Dense 5G deployments can involve thousands of simultaneously connected users. Directly measuring the throughput of every device is expensive and difficult to scale.

This project investigates whether a user's throughput can instead be predicted from measurements already available to the network, including:

- Downlink and uplink SINR
- Physical Resource Blocks (PRB)
- Block Error Rate (BLER)
- Traffic type
- Serving Radio Unit
- User position
- Throughput measurements collected from dedicated operator measurement devices

The target variable is:

```text
throughput_mbps
Dataset

The experiments use the ACC Arena scenario from the:

5G High Density Demand (HDD) Dataset in Liverpool City Region, UK

The original ACC Arena scenario contains:

12,000 users
10,000 timestamps
33 Radio Units
six traffic classes
user position and traffic information
RU association
SINR DL / UL
throughput
PRB utilization
BLER

For computationally tractable experimentation, this project uses a reproducible subset containing:

220 users
108,240 rows
492 samples per user
user IDs 0-219

The subset configuration is:

Operator measurement devices: 20
Measurement user IDs:          0-19

Target users:                  200
Target user IDs:               20-219

Time stride:                   20

The derived dataset is not redistributed in this repository.

See data/README.md for the original dataset source and the exact procedure used to reproduce the local subset.

Operator Measurement Devices

The distinguishing feature-engineering step in this project uses a subset of users as operator measurement devices.

For a given value of X:

users 0 ... X-1

act as network probes.

For every target user, the pipeline finds operator devices that:

are observed at the same timestamp, and
have the same traffic type as the target user.

Their measured throughput is aggregated into:

operator_same_type_count
operator_same_type_mean_throughput
operator_same_type_median_throughput
operator_same_type_max_throughput
operator_same_type_min_throughput
operator_same_type_std_throughput

Conceptually:

Operator measurement devices
            │
            ▼
Group by (timestamp, traffic_type)
            │
            ▼
count / mean / median / min / max / std
            │
            ▼
Network-state features for target users

The operator devices themselves are excluded from the prediction dataset, so their own throughput is never used as the target being predicted.

Feature Set
Radio and network features
sinr_dl
sinr_ul
prb
bler
x
y
z
traffic_type
ru_id
Engineered operator features
operator_same_type_count
operator_same_type_mean_throughput
operator_same_type_median_throughput
operator_same_type_max_throughput
operator_same_type_min_throughput
operator_same_type_std_throughput

Categorical features (traffic_type, ru_id) are one-hot encoded.

Numerical features are standardized for the MLP pipeline.

Preprocessing is implemented through scikit-learn Pipeline and ColumnTransformer objects so transformations are fitted on training data rather than on the complete dataset.

Methodology
Chronological Train/Test Split

A random split would allow future observations from a user to influence predictions for earlier observations.

Instead, the dataset is split chronologically for each user:

oldest 80%  → training
newest 20%  → testing

This preserves temporal ordering and better represents a deployment in which historical measurements are used to predict future network behaviour.

Time-Aware Cross-Validation

Hyperparameter tuning uses:

TimeSeriesSplit

instead of shuffled K-fold cross-validation.

Each validation fold therefore occurs after the data used to train that fold.

Centralized Models

Two model families are evaluated.

Random Forest Regressor

A tree-based ensemble suitable for nonlinear tabular relationships.

Hyperparameters explored include:

n_estimators
max_depth
min_samples_leaf
Multi-Layer Perceptron

A feed-forward neural network trained on the preprocessed tabular feature set.

Hyperparameters explored include:

hidden_layer_sizes
alpha
learning_rate_init

Both models are tuned with GridSearchCV and time-aware cross-validation.

Centralized Results

Results for the main experiment with X = 20:

Model	MAE	RMSE	R²	Training time
Random Forest	0.683	4.581	0.015	58.5 min
MLP	0.661	4.480	0.058	6.2 min

The tuned MLP provides the best centralized trade-off in this experiment:

lower MAE
lower RMSE
higher R²
substantially lower tuning time

Training times were measured in Google Colab and should be interpreted as an indicative computational-cost comparison rather than a hardware-independent benchmark.

Target Distribution

Throughput is strongly skewed.

Key statistics from the experimental subset:

Mean:        0.787 Mbps
Median:      0.000 Mbps
95th pct:    2.080 Mbps
99th pct:   14.368 Mbps
Maximum:   513.936 Mbps
Zero share: 60.8%

More than half of the observations have zero throughput, while a small number of samples contain very large values.

This has an important effect on the metrics:

MAE is dominated by the common low-throughput observations.
RMSE strongly penalizes rare large prediction errors.
R² is particularly affected by the model's inability to explain rare throughput peaks.

For this reason, a relatively low R² does not tell the full story of model behaviour.

Impact of the Number of Operator Devices

The experiment evaluates:

X ∈ {3, 5, 10, 20}

For every value of X, the operator features are rebuilt and the models are evaluated again.

The results are not monotonic.

More operator devices can provide:

better traffic-type coverage
richer measurements of current network conditions
more stable aggregate throughput statistics

However, increasing X also removes more users from the target prediction dataset.

The best value of X therefore depends on both the model and the evaluation metric.

Metric	Best RF	Best MLP
MAE	X = 3	X = 10
RMSE	X = 20	X = 20
R²	X = 3	X = 5

The experiment shows that more measurement devices do not automatically imply better prediction performance.

Federated Learning

The advanced experiment simulates Federated Averaging (FedAvg) across users.

Each eligible user acts as one federated client.

The basic training loop is:

                  Global model
                       │
                       ▼
               Broadcast weights
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       Client 1     Client 2      Client N
          │            │            │
          ▼            ▼            ▼
     Local training Local training Local training
          │            │            │
          └────────────┼────────────┘
                       ▼
              Weighted FedAvg
                       │
                       ▼
               Updated global model

FedAvg computes a weighted average of client model parameters according to the number of training samples available at each client.

Configuration
Federated rounds:       20
Local epochs:            3
Minimum client samples: 50

MLP architecture:
Input → 128 → 64 → Output

Activation: ReLU
Loss:       MSE
Optimizer:  Adam

The federated simulation uses centralized preprocessing followed by decentralized per-user model training.

It therefore models the FedAvg training process, but it should not be interpreted as a fully decentralized end-to-end production system.

Federated Learning Results

The federated experiment compares:

centralized Random Forest
centralized PyTorch MLP
FedAvg global MLP
local-only per-user MLPs
Model	MAE	RMSE	R²
Centralized RF	0.683	4.581	0.015
Centralized MLP (PyTorch)	0.796	4.582	0.015
FedAvg MLP	0.614	4.761	-0.064
Local-only MLP	0.825	4.667	-0.022

FedAvg achieves the lowest MAE, indicating strong performance on the common low-throughput samples.

However, it produces worse RMSE and R².

A likely mechanism is client heterogeneity.

Individual users tend to contain strongly user-specific and traffic-specific behaviour. Averaging specialized local models can smooth out patterns associated with rare high-throughput observations.

As a result:

FedAvg
   ↓
better common-case absolute error
   ↓
lower MAE

but

rare large errors remain
   ↓
higher RMSE
lower R²
Repository Structure
.
├── data/
│   ├── README.md
│   ├── raw/                         # Local dataset, excluded from Git
│   └── sample/
│
├── figures/
│
├── notebooks/
│   └── 01_throughput_prediction_analysis.ipynb
│
├── reports/
│   └── presentation.pdf
│
├── results/
│
├── scripts/
│   └── extract_acc_arena_subset.py
│
├── .gitignore
├── README.md
└── requirements.txt
notebooks/

Contains the complete experimental workflow:

EDA
→ feature engineering
→ preprocessing
→ temporal splitting
→ centralized models
→ X sensitivity analysis
→ Federated Learning
→ evaluation
scripts/

Contains the reproducible dataset extraction pipeline used to convert the original ACC Arena metric-specific CSV shards into the compact tabular dataset consumed by the notebook.

data/

Contains documentation describing how to obtain and reproduce the experimental dataset.

Raw data are intentionally excluded from version control.

reports/

Contains the technical project presentation.

View the project presentation

Reproducing the Project
1. Clone the repository
git clone <https://github.com/garmaar/5g-throughput-prediction.git>
cd 5g-throughput-prediction 

2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

On Windows:

.venv\Scripts\activate
3. Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
4. Obtain the dataset

Follow the instructions in:

data/README.md

The extraction script will generate:

data/raw/acc_arena_subset.csv
5. Run the analysis

Open:

notebooks/01_throughput_prediction_analysis.ipynb

using the Python interpreter from the project's .venv.

The notebook uses repository-relative paths and does not depend on Google Colab or Google Drive.

Limitations

The current implementation has several important limitations:

Experiments use a compact subset rather than the complete 12,000-user ACC Arena dataset.
The throughput distribution is highly skewed, with 60.8% zero-valued observations.
Only four values of X were evaluated.
The centralized comparison uses a tuned scikit-learn MLP, whereas the federated experiment uses a separate PyTorch MLP implementation.
The scikit-learn and PyTorch MLP results should therefore not be interpreted as a direct implementation-to-implementation comparison.
Federated training is performed independently per user, but feature preprocessing is fitted centrally before federated model training.
FedAvg is evaluated as a simulation rather than through physically distributed clients or a production federated-learning framework.

Potential future improvements include:

evaluating the complete ACC Arena dataset
testing a wider range of X
applying log1p or other approaches to the highly skewed throughput target
improving measurement-device traffic-type coverage
tuning the PyTorch architecture and optimization parameters
implementing decentralized or federated preprocessing
investigating methods designed for non-IID federated data
Technologies
Networking
5G New Radio
SINR
PRB
BLER
Radio Unit association
network telemetry
Machine Learning
Python
pandas
NumPy
scikit-learn
Random Forest
Multi-Layer Perceptron
PyTorch
Federated Learning
FedAvg
time-series cross-validation
Engineering
reproducible data extraction
Python virtual environments
Git / GitHub
modular repository structure
experiment reproducibility
Academic Context

This project was developed for Network Data Analysis Laboratory at Politecnico di Milano.

Project assignment:

Throughput Prediction in a Dense 5G Deployment with Federated Learning

Team:

Pablo Garcia
Esteban Lopez
Pietro Limoni

The project assignment required:

throughput regression on ACC Arena users
comparison between a Neural Network and Random Forest
operator measurement devices as additional network-state features
sensitivity analysis over different values of X
Federated Learning using FedAvg
Dataset Citation

Maheshwari, M. K., Raschellà, A., Mackay, M. et al.

5G High Density Demand Dataset in Liverpool City Region, UK.

Scientific Data, 12, 1992 (2025).

DOI: 10.1038/s41597-025-06282-0

Original dataset:

5G High Density Demand (HDD) Dataset in Liverpool City Region, UK

Liverpool John Moores University.

DOI: 10.24377/LJMU.d.00000236

Acknowledgements

The original 5G HDD dataset was produced as part of the Liverpool City Region High Density Demand (LCR HDD) project.

This repository contains the analysis, preprocessing logic, feature engineering, machine-learning experiments, and Federated Learning simulation developed for the academic project. The original third-party dataset is not redistributed here.