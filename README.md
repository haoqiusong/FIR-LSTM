# FIR-LSTM
This is the architecture of FIR-LSTM, a uni-directional LSTM model explained by Layer-wise Relevance Propagation (LRP).

<div align="center">
	<img width="391" alt="image" loc="center" src="https://github.com/user-attachments/assets/40614232-b4e0-40c3-a70a-4d4a9058758b" />
</div>

Architecture of FIR-LSTM. (a). Schematic of the unidirectional LSTM cell and its gating mechanism. The superimposed blue arrows represent the backpropagation of relevance scores via Layer-wise Relevance Propagation (LRP), illustrating how feature importance is determined through back to earlier time steps and features. (b). Overview of the two-layer LSTM model architecture to generate a risk score.


In our experiments, the input is the time-series data, each datapoint contains 6 time epochs, and each time epoch contains 66 features. The final outputs include the binary classification outcome and the calibrated risk score.
