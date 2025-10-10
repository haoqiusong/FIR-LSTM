# FIR-LSTM
FIR-LSTM is an interpretable deep learning framework for clinical time-series prediction.
It uses a uni-directional LSTM model explained by Layer-wise Relevance Propagation (LRP) to identify how each feature and time epoch contributes to the final risk prediction.

<div align="center">
	<img width="535" alt="Screenshot 2025-06-02 at 11 25 06 AM" loc="center" src="https://github.com/user-attachments/assets/ea57ef88-c95d-440b-886e-55c680f2b8de" />
</div>

Architecture of FIR-LSTM. (a). Schematic of the LSTM cell and LRP-based relevance backpropagation.
(b). Two-layer LSTM model generating the risk score.


In our experiments, the input is the time-series data, each datapoint contains 6 time epochs, and each time epoch contains 66 features. The final outputs include the binary classification outcome and the calibrated risk score.
