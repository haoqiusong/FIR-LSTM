# FIR-LSTM
FIR-LSTM is an interpretable deep learning framework for clinical time-series prediction.
It uses a uni-directional LSTM model explained by Layer-wise Relevance Propagation (LRP) to identify how each feature and time epoch contributes to the final risk prediction.

# Model Overview

Architecture: multi-layer unidirectional LSTM

Explainability: LRP backpropagates relevance to quantify feature importance

Inputs: 6 time epochs × 66 features (per patient sequence)

Outputs: Binary classification + calibrated risk score

<div align="center">
	<img width="535" alt="Screenshot 2025-06-02 at 11 25 06 AM" loc="center" src="https://github.com/user-attachments/assets/ea57ef88-c95d-440b-886e-55c680f2b8de" />
</div>

Figure. Architecture of FIR-LSTM. (a). Schematic of the LSTM cell and LRP-based relevance backpropagation.
(b). Two-layer LSTM model generating the risk score.

# Repository Structure

FIR-LSTM/
│
├── benchmark.py                   # Training and evaluation script
├── feature_importance_validation.py   # LRP feature relevance analysis
├── model.py                        # LSTM model definition
├── multi_layers.py                 # Multi-layer LSTM variant
├── LICENSE
└── README.md
