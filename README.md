# CrisprFusion
A sgRNA activity prediction method based on deep learning.

## Dependencies
- Python 3.7+
- TensorFlow == 2.5.0
- NumPy == 1.19.5
- Pandas == 1.1.4

## Quick Start
1. Clone the repository:
   ```bash
   git clone https://github.com/username/CrisprFusion.git
   cd CrisprFusion

## Tested demo with testsets
`python main.py`

## Files and directories description
+ [data](https://github.com/cwk644/CrisprFusion/tree/main/data) splited sgRNA datasets used in this study,saved in numpy for training
+ [data_csv] (https://github.com/cwk644/CrisprFusion/tree/main/Datasets) original datasets saved with csv format
+ [bedfeatures] (https://github.com/cwk644/CrisprFusion/tree/main/bedfeatures) part of biological_features,extracted by bedtools
+ [features] (https://github.com/cwk644/CrisprFusion/tree/main/features) part of biological_features
+ [model] (https://github.com/cwk644/CrisprFusion/tree/main/model) saved model of CrisprFusion
+ [main.py] (https://github.com/cwk644/CrisprFusion/tree/main/main.py) code for model structure and training
+ [load_data_func.py] (https://github.com/cwk644/CrisprFusion/tree/main/load_data_func.py) code for integration of biological_feature

  


