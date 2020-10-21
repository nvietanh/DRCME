This folder contains the code for the paper entitled "Distributionally Robust Local Non-parametric Conditional Estimation"

%%%%%%%%%%%%%%%%%%%%%%

Experiment with synthetic dataset

To replicate the results, one needs to run the ipython notebook Local_Conditional_Estimate_Synthetic.ipynb

Figures in paper:
local_Error.pdf
local_cdf.pdf

%%%%%%%%%%%%%%%%%%%%%%

Experiment with the MNIST dataset 

1) In order to repeat our experiments, one needs to run the python script: 
RunMNIST_LargeTest.py

Results are stored in : LargeTest_results_Euclidean7.dat

2) In order to repeat the analysis, one needs to run the jupyter notebook:
Local_Conditional_Estimate_MNIST.ipynb

Figures presented in the paper and supplementary material will be overwritten:

Figures in paper:
errorCDFs.pdf
adversarialFig1.pdf
adversarialFig2.pdf

Figures in supplementary material:
typePdeviation.pdf
adversarialFigExtra0.pdf
adversarialFigExtra13.pdf
adversarialFigExtraBert0.pdf
adversarialFigExtraBert13.pdf

3) Implementations are all included in:
conditionalEstimates.py

4) All files require the following Python packages to be installed:
tensorflow
numpy
scipy
csv
matplotlib
ot (https://pythonot.github.io/)
pickle
joblib
multiprocessing

%%%%%%%%%%%%%%%%%%%%%%

The folder `results' contains all the graph reported in the paper.
