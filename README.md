# AD_RCA
Companion repository to the paper [Anomaly detection and root cause analysis of time series via the integration of a semantic model for predictive maintenance](https://github.com/Niklas2501/AD_RCA)

## Supplementary Resources
Due to the large file size, all additional supplementary resources including the dataset used for training and evaluation as well as the trained models have been published separately [here](https://www.dropbox.com/sh/yu3hfsnb2rpr2sm/AACiRm7fYi9tBWtl6fQ6RWsga?dl=0).

Detailed results can be found in the subdirectory /logs

## Quick start guide
* Clone this repository
* Replace the data directory in the repository with the one downloaded from the source above
* Add the trained model directories to /data/trained_models if desired
* Change settings in the configuration/Configuration.py script, e.g. which model to execute.
* Use the scripts in /execution to execute the training or testing procedure
