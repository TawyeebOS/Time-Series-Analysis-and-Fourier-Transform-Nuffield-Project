# Time Series Analysis and The Fourier Transform

## Abstract
This is a project for the Nuffield Future Researchers Program. It is intended to act as a basic introduction to time series analysis as well as the fourier transform. Additionally, it is also intended to introduce how the fourier transform can be applied to time series problems by attempting to improve upon a time series classification algorithm. This was achieved by comparing the accuracy of predictions, that the random forest classifier(RFC) algorithm, generated from the FordA data set to the accuracy of the predictions made from the same data set after being processed and transformed using the fourier transform before using the RFC algorithm. The results showed roughly a 20% increase in true values predicted, suggesting that it may be easier to create reliable predictions from time series data once being transformed from the time domain to the frequency domain using the fourier transform.

## Getting started
* download repository (git clone)
* `pip install -r requirements.txt` to install required packages
* run Jupyter lab
* open `Report.ipynb`
