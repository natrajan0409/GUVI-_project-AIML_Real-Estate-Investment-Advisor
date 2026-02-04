# 🏠 Real Estate Investment Advisor

**Predicting Property Profitability & Future Value using Machine Learning**

A comprehensive data science capstone project that analyzes 250,000 properties across India to predict investment viability and future property values using classification and regression models.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Machine Learning Models](#machine-learning-models)
- [Streamlit Application](#streamlit-application)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Future Enhancements](#future-enhancements)
- [Author](#author)

---

## 🎯 Project Overview

This project aims to help real estate investors make data-driven decisions by:

1. **Classification**: Predicting whether a property is a "Good Investment" or "Not Recommended"
2. **Regression**: Estimating property value after 5 years

The project demonstrates end-to-end machine learning pipeline including data preprocessing, exploratory data analysis, model training with MLflow tracking, and deployment via Streamlit web application.

---

## ✨ Features

### Core Functionality
- ✅ **Investment Classification**: Binary classification to identify good investment opportunities
- ✅ **Future Price Prediction**: Regression model to predict 5-year property value
- ✅ **Comprehensive EDA**: 20 research questions answered with visualizations
- ✅ **MLflow Integration**: Experiment tracking and model management
- ✅ **Interactive Web App**: Streamlit-based user interface

### Key Capabilities
- Property price prediction based on location, size, and amenities
- ROI calculation and investment insights
- City-wise and property type analysis
- Interactive data filtering and visualization
- Model performance comparison

---

## 📊 Dataset

**Source**: India Housing Prices Dataset  
**Size**: 250,000 properties  
**Features**: 23 original features + 13 engineered features  
**Coverage**:
- 20 States
- 42 Cities
- 500 Localities

### Key Features
- **Location**: State, City, Locality
- **Property Details**: Type, BHK, Size (sq ft), Age, Floors
- **Pricing**: Current price, Price per sq ft
- **Amenities**: Furnished status, Parking, Amenities list
- **Infrastructure**: Nearby schools, hospitals, public transport
- **Orientation**: Facing direction, Security features

---

## 📁 Project Structure

```
PROJECT2/
├── data/
│   ├── raw/                          # Raw dataset (not in Git)
│   └── processed/                    # Processed data + EDA visualizations
│       ├── housing_data_processed.csv
│       └── q1-q20_*.png             # 20 EDA visualizations
├── src/
│   ├── data_preprocessing.py        # Data cleaning & feature engineering
│   ├── eda_analysis.py              # Exploratory data analysis (20 questions)
│   └── model_training.py            # ML model training with MLflow
├── streamlit_app.py                 # Web application
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation
├── EDA_Summary.md                   # Detailed EDA report
└── .gitignore                       # Git ignore rules
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Setup Instructions

1. **Clone the Repository**
```bash
git clone <your-repository-url>
cd PROJECT2
```

2. **Create Virtual Environment**
```bash
python -m venv venv
```

3. **Activate Virtual Environment**

**Windows:**
```bash
.\venv\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

4. **Install Dependencies**
```bash
pip install -r requirements.txt
```

5. **Download Dataset**
- Place `india_housing_prices.csv` in the project root directory
- Or download from: [Dataset Link]

---

## 💻 Usage

### 1. Data Preprocessing

Process the raw dataset and create engineered features:

```bash
python src/data_preprocessing.py
```

**Output**: `data/processed/housing_data_processed.csv`

**What it does**:
- Handles missing values
- Encodes categorical variables
- Creates target variables:
  - `Good_Investment` (Classification)
  - `Future_Price_5Y` (Regression)
- Generates 36 features from 23 original features

---

### 2. Exploratory Data Analysis

Run comprehensive EDA to answer 20 research questions:

```bash
python src/eda_analysis.py
```

**Output**: 20 visualization files in `data/processed/`

**Analysis Categories**:
- Price & Size Analysis (Q1-5)
- Location-Based Analysis (Q6-10)
- Feature Correlations (Q11-15)
- Investment & Amenities (Q16-20)

---

### 3. Model Training

Train multiple ML models with MLflow tracking:

```bash
python src/model_training.py
```

**Classification Models** (7):
- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier
- Decision Tree Classifier
- Gradient Boosting Classifier
- Naive Bayes
- AdaBoost Classifier

**Regression Models** (6):
- Linear Regression
- Random Forest Regressor
- XGBoost Regressor
- Decision Tree Regressor
- Gradient Boosting Regressor
- Ridge Regression

**Features**:
- Automatic experiment tracking
- Model comparison
- Metric logging (Accuracy, Precision, Recall, F1, ROC AUC, RMSE, MAE, R²)
- Model artifact storage

---

### 4. View MLflow Experiments

Launch MLflow UI to compare model performance:

```bash
mlflow ui
```

Access at: `http://localhost:5000`

**What you can do**:
- Compare all model runs
- View metrics and parameters
- Download trained models
- Analyze experiment history

---

### 5. Launch Streamlit Application

Start the interactive web application:

```bash
streamlit run streamlit_app.py
```

Access at: `http://localhost:8501`

**Application Pages**:
1. **🔮 Predict**: Property investment prediction
2. **📊 Analytics Dashboard**: Interactive data exploration
3. **📈 Market Insights**: City and property type analysis
4. **📉 EDA Results**: All 20 EDA visualizations
5. **ℹ️ About**: Project information

---

## 📈 Exploratory Data Analysis

### 20 Research Questions Answered

#### Price & Size Analysis (Q1-5)
1. Distribution of property prices
2. Distribution of property sizes
3. Price per sq ft by property type
4. Relationship between size and price
5. Outliers in price and size

#### Location-Based Analysis (Q6-10)
6. Average price per sq ft by state
7. Average property price by city
8. Median age of properties by locality
9. BHK distribution across cities
10. Top 5 most expensive localities

#### Feature Relationships (Q11-15)
11. Correlation matrix of numeric features
12. Nearby schools vs price per sq ft
13. Nearby hospitals vs price per sq ft
14. Price by furnished status
15. Price per sq ft by facing direction

#### Investment Analysis (Q16-20)
16. Owner type distribution
17. Availability status distribution
18. Parking space impact on price
19. Amenities impact on price per sq ft
20. Public transport accessibility vs price

**Detailed Report**: See [EDA_Summary.md](EDA_Summary.md)

---

## 🤖 Machine Learning Models

### Classification Task: Good Investment Prediction

**Target Variable**: `Good_Investment` (Binary: 0 or 1)

**Evaluation Metrics**:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC AUC

**Best Model**: [To be updated after training]

---

### Regression Task: Future Price Prediction

**Target Variable**: `Future_Price_5Y` (Continuous)

**Evaluation Metrics**:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score

**Best Model**: [To be updated after training]

---

## 🌐 Streamlit Application

### Features

#### 1. Prediction Page
- Input property details
- Get investment recommendation
- View future price estimate
- Calculate ROI

#### 2. Analytics Dashboard
- Filter by state, BHK, price range
- View summary metrics
- Interactive visualizations
- City-wise analysis

#### 3. Market Insights
- Property type comparison
- Furnished status impact
- Investment opportunities by city
- Key market statistics

#### 4. EDA Results
- All 20 visualizations organized in tabs
- Statistical summaries
- Interactive exploration

---

## 📊 Results

### Dataset Statistics
- **Total Properties**: 250,000
- **Mean Price**: ₹255 Lakhs
- **Mean Size**: 2,748 sq ft
- **Good Investments**: 50.01%
- **Price Range**: ₹10L - ₹500L

### Key Findings
1. **Strong Correlation**: Property size and price (positive)
2. **Location Impact**: Significant price variation across states/cities
3. **Amenities Premium**: Furnished properties command 15-20% higher prices
4. **Infrastructure Value**: Proximity to schools/hospitals increases value
5. **Outliers**: 8.01% properties have exceptional price per sq ft

### Model Performance
[To be updated after model training completion]

---

## 🛠️ Technologies Used

### Programming & Libraries
- **Python 3.14**
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning
- **XGBoost** - Gradient boosting

### Visualization
- **Matplotlib** - Static plots
- **Seaborn** - Statistical visualizations
- **Plotly** - Interactive charts

### ML Operations
- **MLflow** - Experiment tracking
- **Streamlit** - Web application

### Development Tools
- **Jupyter Notebook** - Exploratory analysis
- **Git** - Version control
- **Virtual Environment** - Dependency management

---

## 🔮 Future Enhancements

### Model Improvements
- [ ] Hyperparameter tuning with GridSearchCV
- [ ] Ensemble methods combining multiple models
- [ ] Deep learning models (Neural Networks)
- [ ] Time series analysis for price trends

### Feature Engineering
- [ ] Location-based clustering
- [ ] Price trend indicators
- [ ] Seasonal factors
- [ ] Economic indicators integration

### Application Features
- [ ] User authentication
- [ ] Save favorite properties
- [ ] Property comparison tool
- [ ] PDF report generation
- [ ] Email notifications
- [ ] Mobile app version

### Deployment
- [ ] Deploy to Streamlit Cloud
- [ ] Docker containerization
- [ ] CI/CD pipeline
- [ ] API development
- [ ] Database integration

---

## 👨‍💻 Author

**[Your Name]**
- GUVI Data Science Capstone Project
- February 2026

---

## 📄 License

This project is created for educational purposes as part of the GUVI Data Science course.

---

## 🙏 Acknowledgments

- GUVI for the project opportunity
- Dataset providers
- Open-source community for amazing tools

---

## 📞 Contact

For questions or feedback:
- Email: [your-email@example.com]
- LinkedIn: [Your LinkedIn Profile]
- GitHub: [Your GitHub Profile]

---

## 📝 Project Checklist

- [x] Data preprocessing completed
- [x] EDA - 20 questions answered
- [x] 20 visualizations created
- [x] Model training pipeline created
- [x] MLflow integration implemented
- [x] Streamlit application developed
- [x] Documentation completed
- [x] Code cleaned and organized
- [x] Ready for Git submission

---

**⭐ If you find this project helpful, please give it a star!**

---

*Last Updated: February 4, 2026*
