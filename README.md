# ML4CoAdditives
ML4CoAdditives is a machine learning (ML) based data-driven framework that accelerates the discovery of cobalt electrodeposition additives.
![workflow](https://github.com/ai4chips/ML4CoAdditives/blob/main/workflow.tif)

## Description
Traditional additive development relies on time-consuming trial-and-error experiments, which can hardly keep up with the ever-shrinking feature sizes of chips and the continuously rising performance requirements for interconnects. ML4CoAdditives establishes a new strategy for additive screening, with details available in our paper *Data-Driven Discovery of Novel Additives for Superconformal Cobalt Electrodeposition in 3D Interconnects*. This project includes descriptor generation, feature selection for dimensionality reduction, machine learning models construction, and SHAP analysis for feature importance evaluation.

## Getting Started
These guidelines will help you set up and launch a local copy of the project, supporting your development and testing workflows.
### Prerequisites 
To get the project running, first install the following tool:
[ Anaconda](https://www.anaconda.com/download) or [Miniconda]( https://www.anaconda.com/docs/getting-started/miniconda/install)
### Environment Setup
To execute the code in this repo, initialize your development environment with the steps below. Conda is used for dependency management; build the environment by running these commands:

```
cd ./ML4CoAdditives
conda env create -f environment.yml
conda activate ML4CoAdditives
```
