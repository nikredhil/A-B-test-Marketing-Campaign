# Marketing Campaign A/B Testing Analysis

## Project Overview

This project analyzes the effectiveness of different marketing promotion levels on sales performance using statistical A/B testing methodology. The analysis treats different promotion intensities as treatment and control groups to determine if promotional campaigns significantly impact sales outcomes.

## Dataset

The analysis uses the `WA_MarketingCampaign.csv` dataset containing:

- **548 records** across multiple markets and time periods
- **7 variables**: MarketID, MarketSize, LocationID, AgeOfStore, Promotion, week, SalesInThousands

### Key Variables
- `Promotion`: Promotion intensity levels (1=Low/Baseline, 2=Medium, 3=High)
- `MarketSize`: Market categories (Small, Medium, Large)
- `SalesInThousands`: Sales performance metric
- `AgeOfStore`: Store maturity indicator

## Methodology

### Group Assignment
- **Treatment Group**: Stores with medium (2) or high (3) promotion levels
- **Control Group**: Stores with low/baseline (1) promotion levels

### Analysis Framework
1. **Descriptive Statistics**: Compare average sales between groups
2. **Hypothesis Testing**: Statistical significance testing using simulation
3. **Regression Analysis**: Logistic regression modeling for high sales prediction
4. **Market Segmentation**: Analysis including market size effects

### Regression Results
- Treatment effect on achieving high sales performance
- Market size impact analysis
- Interaction effects between promotion level and market characteristics

## Repository Structure

```
marketing-campaign-analysis/
│
├── data/
│   └── WA_MarketingCampaign.csv
│
├── notebooks/
│   └── marketing_ab_analysis.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── statistical_analysis.py
│   └── visualization.py
│
├── results/
│   ├── figures/
│   └── reports/
│
├── requirements.txt
└── README.md
```

## Installation & Setup

### Prerequisites
```bash
Python 3.8+
Jupyter Notebook
```

### Required Libraries
```bash
pip install pandas numpy scipy statsmodels matplotlib seaborn scikit-learn
```

### Clone Repository
```bash
git clone https://github.com/yourusername/marketing-campaign-analysis.git
cd marketing-campaign-analysis
pip install -r requirements.txt
```

## Usage

### Quick Start
```python
import pandas as pd
import numpy as np
from src.statistical_analysis import MarketingAnalysis

# Load data
df = pd.read_csv('data/WA_MarketingCampaign.csv')

# Initialize analysis
analyzer = MarketingAnalysis(df)

# Run complete analysis
results = analyzer.run_full_analysis()
```

### Jupyter Notebook
Open and run `notebooks/marketing_ab_analysis.ipynb` for step-by-step analysis.

## Analysis Steps

### 1. Data Preprocessing
```python
# Create treatment/control groups
df['group'] = df['Promotion'].map({
    3: 'treatment',  # High promotion
    2: 'treatment',  # Medium promotion  
    1: 'control'     # Low/baseline promotion
})

# Filter for properly aligned records
df2 = df[
    ((df['group'] == 'treatment') & (df['Promotion'].isin([2, 3]))) |
    ((df['group'] == 'control') & (df['Promotion'] == 1))
]
```

### 2. Statistical Testing
```python
# Hypothesis Test
# H0: sales_treatment - sales_control <= 0
# H1: sales_treatment - sales_control > 0

# Calculate observed difference
obs_diff = treatment_avg_sales - control_avg_sales

# Simulate null distribution
null_diffs = simulate_null_distribution(n_simulations=10000)

# Calculate p-value
p_value = (null_diffs >= obs_diff).mean()
```

### 3. Regression Analysis
```python
# Logistic regression for high sales prediction
from sklearn.linear_model import LogisticRegression

# Create binary outcome (above/below median sales)
df2['high_sales'] = (df2['SalesInThousands'] > median_sales).astype(int)

# Fit model
model = LogisticRegression()
model.fit(X_train, y_train)
```

## Key Metrics

### Performance Indicators
- **Sales Lift**: Percentage improvement in treatment vs control
- **Statistical Significance**: P-value from hypothesis testing
- **Effect Size**: Magnitude of treatment impact
- **Model Accuracy**: Prediction performance metrics

### Visualizations
- Sales distribution by group
- Null distribution vs observed difference
- ROC curve for prediction model
- Market size interaction effects

## Business Implications

### Recommendations
Based on the analysis results:

1. **Promotion Effectiveness**: Data-driven recommendation on promotion ROI
2. **Market Segmentation**: Insights on which market sizes benefit most
3. **Resource Allocation**: Optimization suggestions for promotion budgets
4. **Future Testing**: Recommendations for additional experiments

## Technical Details

### Statistical Methods
- **Two-sample t-test**: For mean difference testing
- **Bootstrap simulation**: For p-value calculation
- **Logistic regression**: For probability modeling
- **Interaction analysis**: For market segmentation insights

### Model Validation
- Train/test split validation
- Cross-validation for robustness
- ROC curve analysis
- Confusion matrix evaluation

## Code Structure

### Main Analysis Pipeline
```python
# Complete analysis workflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, auc

# 1. Load and explore data
df = pd.read_csv('WA_MarketingCampaign.csv')

# 2. Create treatment/control groups
df['group'] = df['Promotion'].map({3: 'treatment', 2: 'treatment', 1: 'control'})

# 3. Statistical testing
treatment_avg = df[df['group'] == 'treatment']['SalesInThousands'].mean()
control_avg = df[df['group'] == 'control']['SalesInThousands'].mean()

# 4. Hypothesis testing with simulation
# 5. Logistic regression modeling
# 6. Results interpretation
```

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/analysis-improvement`)
3. Commit changes (`git commit -am 'Add new analysis feature'`)
4. Push to branch (`git push origin feature/analysis-improvement`)
5. Create Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Acknowledgments

- Dataset source: Marketing campaign performance data
- Statistical methodology adapted from A/B testing best practices
- Inspiration from marketing analytics and experimental design literature

---

