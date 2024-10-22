# Mini - Project 1
## Group Number 22

Team Members:

- Hardik Jindal (220420) 

- Naman Gupta (220686)

- Gupil (220416) 

- Trijal Srivastava (221144) 

- Harshit Srivastava (220444) 


## Overview
This project involves the classification of test data from three different datasets: emoticons, deep features, and text sequences. Each dataset has a unique representation of the input data, requiring specific preprocessing and model loading techniques. The final goal is to combine the predictions from these datasets for a comprehensive prediction.

## Project Structure
The project consists of the following sections:
1. Dataset 1: Emoticon-Based Classification
2. Dataset 2: Deep Feature-Based Classification
3. Dataset 3: Text Sequence-Based Classification
4. Combined Prediction: Merging all datasets for a unified prediction

## Prerequisites

Install the required libraries by:
```bash
pip install -r requirements.txt
```

## Directory Structure
The directory structure is as follows:
```
├── datasets
│   ├── test
│   │   ├── test_emoticon.csv
│   │   ├── test_feature.npz
│   │   └── test_text_seq.csv
│   ├── train
│   │   ├── train_emoticon.csv
│   │   ├── train_feature.npz
│   │   └── train_text_seq.csv
│   └── valid
│       ├── valid_emoticon.csv
│       ├── valid_feature.npz
│       └── valid_text_seq.csv
├── read_data.py
├── README.md
├── models
│    ├── dataset1.keras
|    ├── dataset2.joblib
|    ├── dataset3.keras
|    └── combined.joblib
├── 22.py
├── combined.ipynb
├── dataset1.ipynb
├── dataset2.ipynb
├── dataset3.ipynb
├── README.md
├── pred_emoticon.txt
├── pred_deepfeat.txt
├── pred_textseq.txt
├── pred_combined.txt
└── requirements.txt
```
## Generating Test predictions
In order to generate the test predictions for all the 4 required models, simply run `22.py` after loading the datasets at the path mentioned in the directory structure above. Alternately, you can change the file path in `22.py`.

The .txt files will automatically be generated.

## Output Files
The predictions for each dataset are saved in the following files:
- `pred_emoticon.txt`: Predictions for Dataset 1 (emoticon-based).
- `pred_deepfeat.txt`: Predictions for Dataset 2 (deep feature-based).
- `pred_textseq.txt`: Predictions for Dataset 3 (text sequence-based).
- `pred_combined.txt`: Combined predictions for all datasets.

## Instructions
1. Ensure all required datasets and models are placed in the correct directories as outlined.
2. Run the script to generate predictions for each dataset and the combined model.
3. The predictions will be saved in the respective `.txt` files.

## Notes
- Make sure to preprocess data consistently for all datasets.
- Adjust model paths if they differ from the structure described.
- This project assumes that the models were trained separately and are ready to be used for prediction.
