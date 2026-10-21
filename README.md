# Dementia Speech Screening

A machine learning project that estimates Alzheimer's/dementia risk from speech, using linguistic features extracted from transcripts in the DementiaBank dataset. The work was published as a research paper covering the results and the ethics of AI-based health screening.

## Overview

Changes in language are an early sign of cognitive decline. People at risk often use a smaller vocabulary, pause more, and struggle to find words. This project turns those patterns into numeric features and trains classifiers to flag speech that may indicate dementia.

## Approach

1. **Data:** speech samples from the DementiaBank dataset.
2. **Feature extraction:** linguistic features such as vocabulary richness and pause patterns.
3. **Preprocessing:** feature scaling with a saved scaler, so new samples are transformed the same way as the training data.
4. **Modelling:** Logistic Regression and Gradient Boosting classifiers built with scikit-learn.
5. **Saving:** trained models and scalers are stored as `.pkl` files so they can be reloaded without retraining.

## Tech stack

Python, scikit-learn, NLP, Jupyter Notebook

## Repository contents

| File | Purpose |
|---|---|
| `alzehimer.ipynb` | Main notebook: feature extraction, training and evaluation |
| `dementia_logistic_model.pkl` | Trained Logistic Regression model |
| `dementia_gradient_boosting_model.pkl`, `dementia_gb_model.pkl`, `gradient_boosting_dementia_model.pkl` | Trained Gradient Boosting models |
| `feature_scaler.pkl`, `gb_scaler.pkl` | Fitted scalers for the input features |
| `model_metadata.json`, `gb_metadata.json` | Metadata describing the saved models |

## Running the project

    pip install numpy pandas scikit-learn jupyter
    jupyter notebook alzehimer.ipynb

To reuse a saved model:

    import joblib
    model = joblib.load("dementia_gradient_boosting_model.pkl")
    scaler = joblib.load("feature_scaler.pkl")

## Data

The DementiaBank data is not included in this repository. It is available from DementiaBank (TalkBank) under their access terms.

## Ethics and limitations

This project is a research prototype, **not a diagnostic tool**. A screening model can produce false positives, which may cause distress, and false negatives, which may give false reassurance. Results depend on the dataset, which may not represent all accents, languages, ages or recording conditions. Any real-world use would need clinical validation and professional oversight. These ethical questions are discussed in the accompanying research paper.

## Author

Aradhana Shukla
