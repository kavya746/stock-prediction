
# Stock Market Movement Prediction

## Overview

This project is a machine learning-based stock market prediction model. It predicts whether the market will go up or down on the next day based on historical stock data. The focus is on predicting the movement of the **S\&P 500 index**. The model uses **Random Forest Classifier** to make predictions and evaluates them with precision scores.

## Objective

The objective of this project is to predict whether the stock market will close higher or lower on the next day based on historical data. The **target** for prediction is binary:

* `1` if the price will increase the next day.
* `0` if the price will decrease or stay the same.

## Data Collection and Processing

1. **Data Source**:
   The historical data for the S\&P 500 index is fetched using the `yfinance` library. If the data is not available locally (`sp500.csv`), it is downloaded and saved for future use.

2. **Data Format**:
   The data consists of the following columns:

   * `Open`, `High`, `Low`, `Close`, `Volume`, `Dividends`, and `Stock Splits`
     The date is set as the index for better time-based operations.

3. **Data Preprocessing**:

   * **Date Parsing**: The date index is converted to a proper `datetime` format for easier handling.
   * **Feature Engineering**:

     * **Tomorrow’s Closing Price**: A new column is added which holds the closing price for the next day (`Tomorrow`).
     * **Target Variable**: The target column is created, where `1` represents the price going up the next day (`Tomorrow > Close`), and `0` otherwise.
   * **Feature Selection**: Columns such as `Dividends` and `Stock Splits` are removed, as they don’t provide meaningful information for this prediction task.

4. **Data Splitting**:
   The dataset is split into a training set (starting from 1990) and a testing set (last 100 rows). This ensures that the model is trained on historical data and validated on unseen data.

## Model Building and Training

1. **Model Selection**:
   The project uses the **Random Forest Classifier** algorithm. This is an ensemble learning method that combines multiple decision trees to improve prediction accuracy and robustness.

   * The model is trained with 100 trees (`n_estimators=100`) and a minimum sample split size of 100 (`min_samples_split=100`) to avoid overfitting.

2. **Training and Testing**:

   * The model is trained on the selected features: `Close`, `Volume`, `Open`, `High`, and `Low`.
   * After training, predictions are made for the test set.
   * A **precision score** is calculated to evaluate the performance of the model on the test set. Precision measures the proportion of positive predictions that are actually correct.

## Backtesting

1. **Backtesting**:
   The model is evaluated over multiple periods in a backtesting framework, where the model is trained incrementally (with increasing training data size) and tested on subsequent chunks of data.

   * The data is split into chunks, and for each chunk, the model makes predictions.
   * The results of these predictions are combined into a final prediction dataset, and the precision score is calculated.

2. **Rolling Window Features**:
   In addition to the basic features, new features are created using rolling windows:

   * **Rolling Averages**: The model includes features that capture trends, such as the rolling average over various periods (e.g., 2, 5, 60, 250, and 1000 days). These features provide insight into the stock's performance over time.
   * **Trend Features**: Trends based on the shift in stock price and the number of days the price increased within these periods are also included.

3. **Prediction Threshold Adjustment**:
   The model predicts probabilities (`predict_proba`) rather than direct class labels. A threshold of 0.6 is applied, where:

   * Probabilities greater than or equal to 0.6 are classified as `1` (price will go up).
   * Probabilities less than 0.6 are classified as `0` (price will stay the same or go down).

## Model Evaluation

1. **Precision Score**:
   The precision score is computed after backtesting, which measures the proportion of true positive predictions among all positive predictions. The model achieved a precision score of approximately **53%**.

2. **Class Distribution**:
   The target classes (up or down) are imbalanced, with more instances of the price going up than down. The model's output distribution reflects this imbalance.

3. **Rolling Averages and Trend Impact**:
   Adding features like rolling averages and trend information significantly enhanced model performance. These features allow the model to understand short-term and long-term market trends.

## Key Features and Engineering

* **Moving Averages**: Various moving averages (2, 5, 60, 250, and 1000 days) are computed for the `Close` price to provide an understanding of short and long-term trends.
* **Trend Analysis**: The trend over these rolling windows is computed to help the model capture momentum in the market.
* **Price Ratios**: The ratio of the current `Close` price to the moving averages is calculated, helping the model understand whether the stock price is above or below its historical average.

## Conclusion

This project provides a good foundation for predicting stock market movements based on historical data and engineered features. The **Random Forest Classifier** model demonstrates decent performance, with further potential for improvement by fine-tuning hyperparameters, adding more features, or using other machine learning algorithms.

