# Tech Giants Stock Analysis (2006-2018)

## 📊 Project Overview

This project analyzes the historical stock price data of four major tech companies (Google, Microsoft, IBM, and Amazon) from 2006 to 2018. Using advanced time series analysis techniques and deep learning models (GRU - Gated Recurrent Units), the project aims to understand market patterns, visualize key trading metrics, evaluate price trends, and build predictive models for stock prices.

## 🎯 Key Objectives

1. **Market Pattern Analysis**: Evaluate the distribution between opening and closing prices, followed by calculating their statistical correlation.
2. **Key Indicator Visualization**: Create comprehensive visual representations of all critical trading metrics (Open, High, Low, Close, Volume).
3. **Price Ceiling Evaluation**: Conduct comparative analysis between peak (High) and settlement (Close) prices across all datasets.
4. **Temporal Pattern Recognition**: Identify and extract underlying trends and seasonal patterns within the historical price data.
5. **Predictive Modeling**: Implement GRU neural networks to predict future stock prices based on historical patterns.

## 📋 Dataset

The project uses four historical stock price datasets spanning from January 2006 to December 2018:

- `GOOGL_2006-01-01_to_2018-01-01.csv`
- `MSFT_2006-01-01_to_2018-01-01.csv`
- `IBM_2006-01-01_to_2018-01-01.csv`
- `AMZN_2006-01-01_to_2018-01-01.csv`

Each dataset contains daily trading information with the following columns:
- Date
- Open
- High
- Low
- Close
- Volume
- Name (Stock ticker)

## 🧰 Technologies & Tools

- **Python**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib & Seaborn**: Basic data visualization
- **Plotly Express**: Interactive visualization
- **PyTorch**: Deep learning framework for GRU models
- **scikit-learn**: Model evaluation and data preprocessing
- **statsmodels**: Time series analysis and seasonal decomposition

## 📈 Analysis Components

### 1. Exploratory Data Analysis
- Summary statistics for each stock
- Checking for missing values
- Visualizing price distributions

### 2. Correlation Analysis
- Correlation between opening and closing prices
- Correlation matrices for all trading metrics
- Volume vs. price correlation analysis

### 3. Temporal Analysis
- Trend identification
- Seasonal decomposition
- Volatility analysis

### 4. Predictive Modeling with GRU
- Data preprocessing and normalization
- Time series sequence creation
- GRU model architecture
- Training and testing methodology
- Performance evaluation (RMSE)

## 📊 Key Findings

### Stock Performance (2006-2018)
- **Amazon**: Most dramatic growth, particularly from 2015-2018, reaching the highest valuation by 2018 (~$1,200)
- **Google**: Strong steady growth with periodic volatility, maintaining second position by 2018 (~$1,100)
- **IBM**: Moderate growth until 2013 followed by gradual decline, stabilizing around $150-200
- **Microsoft**: Lowest initial valuation with minimal growth until 2013, then consistent upward trend to ~$100

### Model Performance
- GRU models achieved varying degrees of success across different stocks
- IBM predictions showed the best performance with a test RMSE of 1.93
- Microsoft predictions achieved a training RMSE of 0.55 but had higher test RMSE of 2.94
- Google and Amazon models showed promising training performance but exhibited signs of overfitting

## 💻 Usage

### Prerequisites
```
Python 3.7+
PyTorch
pandas
numpy
matplotlib
seaborn
plotly
scikit-learn
statsmodels
```

### Installation
```bash
# Clone the repository
git clone https://github.com/username/tech-stock-analysis.git
cd tech-stock-analysis

# Install dependencies
pip install -r requirements.txt
```

### Running the Analysis
```bash
# Run the main analysis notebook
jupyter notebook stock_analysis.ipynb
```

## 📂 Project Structure

```
tech-stock-analysis/
│
├── data/
│   ├── AMZN_2006-01-01_to_2018-01-01.csv
│   ├── GOOGL_2006-01-01_to_2018-01-01.csv
│   ├── IBM_2006-01-01_to_2018-01-01.csv
│   └── MSFT_2006-01-01_to_2018-01-01.csv
│
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   ├── correlation_analysis.ipynb
│   ├── seasonal_decomposition.ipynb
│   └── gru_modeling.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── visualization.py
│   ├── gru_model.py
│   └── utils.py
│
├── results/
│   ├── figures/
│   └── model_checkpoints/
│
├── requirements.txt
└── README.md
```

## 🔮 Future Work

- Incorporate fundamental analysis metrics alongside technical indicators
- Implement additional deep learning models (LSTM, Transformer) for comparison
- Extend the analysis to include more recent data (2018-present)
- Add sentiment analysis from news and social media as additional features
- Develop an ensemble model approach combining different prediction strategies
- Create a web dashboard for interactive exploration of the analysis

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
