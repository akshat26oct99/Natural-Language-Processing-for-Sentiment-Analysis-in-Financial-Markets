# Natural Language Processing for Sentiment Analysis in Financial Markets

This project aims to categorize and analyze the sentiment of financial news articles and tweets using Natural Language Processing (NLP) techniques. It involves training models to classify financial news into predefined categories and assess the sentiment of the text.

## Table of Contents

- [Overview](#overview)
- [Datasets](#datasets)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results](#results)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [References](#references)

## Overview

Financial sentiment analysis is crucial for understanding market trends, investor sentiment, and making informed decisions. This project focuses on two main tasks:
1. **Category Tagger:** Classifying financial news and tweets into 20 predefined categories such as "Analyst Update," "Earnings," "Stock Movement," etc.
2. **Sentiment Tagger:** Assigning sentiment labels (positive, negative, neutral) to financial text.

The project employs both baseline models (e.g., Logistic Regression on TF-IDF vectors) and advanced models like FinBERT, which is pre-trained on financial data.

## Datasets

The project utilizes two primary datasets:
1. **Financial Phrasebank:** Contains human-annotated financial sentences with sentiment labels (positive, negative, neutral).
2. **Twitter Financial News:** Contains finance-related tweets annotated with 20 topic labels.

Datasets can be accessed and loaded using the Hugging Face `datasets` library:
- [Financial Phrasebank](https://huggingface.co/datasets/financial_phrasebank)
- [Twitter Financial News Topic](https://huggingface.co/datasets/zeroshot/twitter-financial-news-topic)


## Installation

1. **Clone the repository** from GitHub to your local machine.
2. **Create a virtual environment** and install the necessary dependencies listed in `requirements.txt`.
3. **Install NLTK dependencies** for text processing.

## Usage

- **Preprocessing:** Run the preprocessing script to clean the data.
- **Training Baseline Model:** Train a Logistic Regression model on TF-IDF vectors of the financial text.
- **Training Advanced Model (FinBERT):** Fine-tune the FinBERT model on the financial dataset.
- **Evaluation:** Evaluate the trained models to assess their performance.

## Model Training and Evaluation

- **Logistic Regression on TF-IDF Vectors:** Achieved an accuracy of approximately 74%. It struggled with identifying positive and negative sentiments due to limited understanding of financial jargon.
- **Random Forest Classifier:** Improved accuracy to about 76%, but still faced limitations with complex financial language.
- **FinBERT Model:** Provided superior results in understanding financial text, outperforming baseline models significantly.

## Results

- The baseline models showed satisfactory results but were limited by their ability to handle financial jargon.
- The FinBERT model, with its specialized training, demonstrated better performance in identifying nuanced sentiments in financial text.

## Limitations

- **Data Imbalance:** Some categories are underrepresented, leading to biased predictions.
- **Domain Specificity:** Baseline models have difficulty with financial terminology, impacting their effectiveness.

## Future Work

- **Data Augmentation:** Explore methods to balance the dataset and enhance model robustness.
- **Advanced Models:** Further fine-tune FinBERT and experiment with other advanced models like GPT-3 or T5.
- **Real-Time Analysis:** Develop a real-time financial sentiment analysis tool for dynamic market insights.

## References

- [FinBERT Model](https://huggingface.co/ProsusAI/finbert)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- Araci, D. (2019). *FinBERT: Financial Sentiment Analysis with Pre-trained Language Models*.


