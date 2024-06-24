# Data mining techniques on astronomical spectra data. IV : Time Domain Analysis

This is the experiment code of the paper -
The codes are written in Python. The dependent packages for the codes include sklearn, statsmodels, glob, numpy, pandas, time, warnings, torch, matplotlib, and scipy. Each algorithm is organized as follows: 1) load training and testing datasets; 2) configure the parameters of the time series prediction models; 3) train the models on the training datasets; 4) make predictions on both the training and testing datasets using the trained models; 5) evaluate the performance of the trained models.

The codes load data from *.csv files, which store tabular data in text format. Each CSV file contains a complete light curve data. These files can be directly read using the pandas library in Python.

For ARIMA, parameter selection is primarily achieved using the Akaike Information Criterion (AIC). We define a function, best arima model, which identifies the combination of ARIMA model parameters (p, d, q) that yields the smallest AIC value by iterating through possible parameter combinations. In our experiments, we set the maximum values of p, d, and q to 5, 2, and 5, respectively. In practical applications, the Bayesian Information Criterion (BIC) and other methods can also be used to determine the optimal parameter combination.

SVR needs to select kernel functions. There are many kernel functions: linear kernel function, polynomial kernel function, RBF kernel function, sigmoid kernel function, etc. Different kernel functions are suitable for different situations. For example, the linear kernel function fails to converge on the delta-scuti dataset.

In the CNN, a dropout layer is applied after the fully connected layer (\texttt{self.fc1}) with a dropout rate set to 0.5. This means that during training, 50\% of the neurons are randomly ignored (i.e., set to 0) at each parameter update, helping to prevent overfitting and improve the model's generalization capability. In RNN-based models, the dropout layer is applied after the outputs of the RNN, GRU, and LSTM layers, but before the fully connected layer.
