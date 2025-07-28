# 📈 FinBERT:-FINANCIAL NEWS SENTIMENTAL ANALYSIS USING DistilBERT 


This project predicts stock market movement based on sentiment analysis of financial news articles collected from **Finviz**, using **BERT** and **DistilBERT** models. It is tailored for stocks from the **NASDAQ** (USA) and **NSE** (India) to help traders make informed decisions with higher accuracy.

---

## 🔍 Problem Statement

The stock market is heavily influenced by news sentiment. Positive or negative news can impact stock prices drastically. This project uses **NLP** and **transformer models** to analyze financial news and predict whether a stock will go **Up** 📈 or **Down** 📉.

---

## 💡 Objectives

- Collect and clean financial news from **Finviz**.
- Perform **sentiment analysis** using `BERT` and `DistilBERT`.
- Classify sentiment to predict stock movement.
- Evaluate models using `Accuracy`, `F1 Score`, `Precision`, and `Recall`.
- Provide useful insights for **NASDAQ** and **NSE** traders.

---

## 🗞️ Data Source

- **News Headlines**: Scraped from [https://finviz.com/](https://finviz.com/)
- **Stocks Covered**: NASDAQ & NSE-listed companies.
- **Labels**: Stock movement direction (Up / Down) based on post-news stock performance.

---

## 🧰 Tech Stack

- Python 🐍
- Transformers (🤗 `BERT`, `DistilBERT`)
- BeautifulSoup, Requests (for scraping Finviz)
- Pandas, NumPy (data wrangling)
- Scikit-learn (metrics & train-test split)
- Matplotlib, Seaborn (visualization)

---

## 📊 Evaluation Metrics

| Metric     | Description                                      |
|------------|--------------------------------------------------|
| Accuracy   | Overall correctness of predictions               |
| Precision  | TP / (TP + FP) – How many selected are relevant? |
| Recall     | TP / (TP + FN) – How many relevant are selected? |
| F1 Score   | Harmonic mean of Precision and Recall            |

---

## 🔁 Workflow

1. **Scrape news** from Finviz using `BeautifulSoup`.
2. **Clean text**: remove stop words, punctuation, etc.
3. **Label data**: based on stock price movement after news.
4. **Sentiment Classification**:
    - Use pre-trained BERT and DistilBERT models from `transformers`.
5. **Train/Test Split**
6. **Evaluate** using `Accuracy`, `Precision`, `Recall`, `F1 Score`
7. **Predict** stock movement: 📈 Up / 📉 Down
8. **Visualize** results with confusion matrix and metric plots.

---

## 🧪 Model Comparison

| Model       | Accuracy | Precision | Recall | F1 Score |
|-------------|----------|-----------|--------|----------|
| BERT        | XX%      | XX%       | XX%    | XX%      |
| DistilBERT  | XX%      | XX%       | XX%    | XX%      |

> Replace `XX%` with your actual results after training.

---

## 📁 Project Structure

