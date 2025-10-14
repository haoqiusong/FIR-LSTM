# FIR-LSTM

FIR-LSTM is an interpretable deep learning framework for clinical time-series prediction.
It uses a uni-directional LSTM model explained by Layer-wise Relevance Propagation (LRP) to identify how each feature and time epoch contributes to the final risk prediction.

## Model Overview

**Architecture**: multi-layer unidirectional LSTM

**Explainability**: LRP backpropagates relevance to quantify feature importance

**Inputs**: 6 time epochs × 66 features (per patient sequence)

**Outputs**: Binary classification performance (ROC, PRC, calibration curve, training loss, feature importance visualizations) + calibrated risk score (for individual datapoints)

<div align="center">
	<img width="535" alt="Screenshot 2025-06-02 at 11 25 06 AM" loc="center" src="https://github.com/user-attachments/assets/ea57ef88-c95d-440b-886e-55c680f2b8de" />
	<p><b>Figure 1.</b> Architecture of FIR-LSTM. (a). Schematic of the LSTM cell and LRP-based relevance backpropagation. (b). Two-layer LSTM model generating the risk score.</p>
</div>

## Repository Structure

```
FIR-LSTM/
│
├── benchmark.py                  		# Model benchmarking script
├── feature_importance_validation.py	# LRP feature importance validation analysis
├── model.py							# FIR-LSTM model definition and training
├── multi_layers.py						# Multi-layer FIR-LSTM variant
├── demo								# Demo run (simulated dataset)
├── LICENSE
└── README.md
```

## Installation

### Clone the repository

```
git clone https://github.com/haoqiusong/FIR-LSTM.git
cd FIR-LSTM
```

### Create a Python environment (optional)

```
conda create -n firlstm python=3.11
conda activate firlstm
```

#### Notes

1. Tested on Python 3.11.5 and PyTorch 2.5.1 (CUDA 11.8).
2. GPU acceleration is automatically used if CUDA is available.
3. You can modify hyperparameters and file paths directly in the Python scripts.
4. The demo, which operates on simulated data rather than real patient records, executes in roughly one minute on a standard desktop workstation.

## Citation

Zhang, L., Song, H., Patel, A., Pollack, M., & Watson, L. (2025). FIR-LSTM: An Explainable Deep Learning Framework for Predicting Iatrogenic Withdrawal Syndrome in Pediatric Intensive Care Units.

## Contact

Haoqiu Song

Ph.D. Candidate, Virginia Tech

📧 haoqiu@vt.edu
