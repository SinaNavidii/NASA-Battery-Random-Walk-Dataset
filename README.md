# NASA Random Walk Dataset Preprocessing

This repository contains preprocessing scripts for the [NASA Random Walk Dataset] (https://data.nasa.gov/Raw-Data/Randomized-Battery-Usage-1-Random-Walk/ugxu-9kjx/about_data)
.

### Methodology

The preprocessing approach follows methodologies proposed in the following papers:

1. **Richardson et al.** (DOI: [10.1016/j.est.2019.03.022](https://doi.org/10.1016/j.est.2019.03.022))  
   *"Battery health prediction under generalized conditions using a Gaussian process transition model"*

2. **Wen et al.** (DOI: [10.1109/TIV.2023.3315548](https://doi.org/10.1109/TIV.2023.3315548))  
   *"Physics-Informed Neural Networks for Prognostics and Health Management of Lithium-Ion Batteries"*

### Overview

The preprocessing scripts included in this repository enable the preparation of the NASA Random Walk Dataset for subsequent modeling and analysis. The main tasks performed during preprocessing include:

- **Data cleaning and filtering** 
- **Feature extraction** based on key battery metrics used in each paper.

### Note:
Customize the directory paths in the preprocessing scripts to match the locations of your dataset and other required files. The paths are located in the configuration section of the scripts and must be updated for the code to function properly.

