# ML-based Rainfall Estimator

ML-based Rainfall Estimator [1] is a machine learning-based tool, developed in python, to estimate the rainfall in the areas in which no rain gauge data is available.  In practice, given as input the concatenation of the data measured by the three data sources (rain gauges, radars and meteosat satellites) at a given time t, our tool returns the estimated class c of the rainfall event and some evaluation measures (i.e., CSI, FAR, POD, etc.).

## Author

The code is developed and maintained by Massimo Guarascio and Pietro Sabatino (massimo.guarascio@icar.cnr.it, pietro.sabatino@icar.cnr.it)

## Usage

First, download this repo:
- You need to have 'python3' installed.
- You also need to install 'numpy', 'matplotplib', 'pandas', 'sklearn' and 'mlxtend'.
- You may also want to install jupyter notebook to run notebook file.

Then, you can run:

python ml_based_rainfall_estimator.py "/yourpath/datasets" training.csv test.csv 0

The last parameter can be set to 1 for verbose debugging.

## References

[1] M. Guarascio, G. Folino, F. Chiaravalloti, S. Gabriele, A. Procopio and P. Sabatino, "A Machine Learning Approach for Rainfall Estimation Integrating Heterogeneous Data Sources", in IEEE Transactions on Geoscience and Remote Sensing, doi: https://doi.org/10.1109/TGRS.2020.3037776.
