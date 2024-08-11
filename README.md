# Movie Review Classifier with Naive Bayes
## Overview
This project implements a movie review classification system using the Naive Bayes algorithm. The goal is to classify movie reviews as either positive or negative based on their content. The project includes data preprocessing, model training, and evaluation of the classifier's performance.

## Project Details
#### Dataset: A collection of movie reviews labeled as positive or negative.
#### Tools & Libraries: Python, scikit-learn, Pandas, NumPy
#### Objective: To build a text classification model that can accurately predict the sentiment of movie reviews.
#### To checkout you can visit-https://movie-review-classifier-nsim.onrender.com/

## Features
#### Data Preprocessing: Includes text cleaning, tokenization, and vectorization of reviews.
#### Model Training: Uses the Naive Bayes algorithm for text classification.
#### Evaluation: Assesses model performance using metrics such as accuracy, precision, recall, and F1-score.

## Installation
#### Navigate to the project directory:
cd movie-review-classifier

#### Install the required dependencies:
pip install -r requirements.txt

## Usage
Prepare your dataset by placing it in the data/ directory. Ensure it is in a format that can be read by the preprocess_data.py script.
#### Run the preprocessing script to clean and vectorize the data:
python preprocess_data.py

#### Train the Naive Bayes model:
python train_model.py

#### Evaluate the model's performance:
python evaluate_model.py

#### Use the trained model to classify new movie reviews:
python predict_review.py --review "Your movie review text here"

## Project Structure
#### preprocess_data.py: Script for data cleaning and vectorization.
#### train_model.py: Script for training the Naive Bayes classifier.
#### evaluate_model.py: Script for evaluating the model's performance.
#### predict_review.py: Script for classifying new movie reviews.
#### data/: Directory for storing raw and preprocessed data.
#### requirements.txt: List of dependencies for the project.

## Results
#### Model Accuracy: Evaluated on the test set to determine the classification accuracy.
#### Performance Metrics: Includes precision, recall, and F1-score to assess model effectiveness.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue if you have suggestions or find bugs.
