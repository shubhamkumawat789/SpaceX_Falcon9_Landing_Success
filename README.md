# 🚀 SpaceX Falcon 9 First Stage Landing Prediction
This project aims to predict whether the Falcon 9 rocket's first stage will land successfully after launch. Using data from SpaceX's API and Wikipedia, we analyze historical launch data to build a classification model that can determine landing outcomes.

## 🎯 Objective
To build a machine learning model that can predict if the Falcon 9's first stage will land successfully based on features like:

### FEATURE DESCRIPTIONS

| Feature Name | Description |
|--------------|-------------|
| LaunchSite | Location where the rocket was launched (e.g., CCAFS SLC 40, KSC LC 39A, VAFB SLC 4E) |
| Orbit | Destination orbit of the payload (e.g., LEO, GTO, ISS, PO, SSO) |
| BoosterVersion | Specific model or variant of the Falcon 9 booster |
| ReuseHistory | Indicates whether the booster was reused from previous flights |
| FlightNumber | Sequential number of the launch mission |
| PayloadMass | Mass of the payload carried by the rocket (in kg) |
| Flights | Number of times the booster has flown before the current mission |
| Block | Booster block version (e.g., Block 1, Block 5) |
| GridFins | Whether grid fins were used for steering during descent (1 = Yes, 0 = No) |
| Legs | Whether landing legs were deployed (1 = Yes, 0 = No) |

### TARGET VARIABLE

| Feature | Description |
|---------|-------------|
| Class | Landing outcome of the booster (1 = Successful landing, 0 = Failed landing) |

### ADDITIONAL FEATURES IN DATASET

| Feature Name | Description |
|--------------|-------------|
| Date | Launch date of the mission |
| ReusedCount | Number of times the booster has been reused |
| Serial | Unique serial number of the booster |
| Longitude | Longitude coordinate of the launch site |
| Latitude | Latitude coordinate of the launch site |
| Outcome | Raw textual description of the landing outcome |
| LandingPad | Type of landing pad used (ground pad or drone ship) |

### FEATURE IMPORTANCE 

| Feature | Reason for Importance |
|---------|-----------------------|
| PayloadMass | Heavy payloads reduce fuel margin |
| BoosterVersion | Newer versions have improved landing technology |
| ReuseHistory | Indicates reliability from previous missions |
| Orbit | Different orbits need different landing strategies |
| GridFins & Legs | Essential for controlled descent and landing |


## 📊 Process Breakdown (SpaceX Falcon 9 first stage Landing.ipynb - Final notebook saved for machine learning modeling)

### Step 1. Data Collection
- **From SpaceX API:** Launch details, rocket info, payload data, core data, launchpad info.
- **From Wikipedia:** Scraped Falcon 9 and Falcon Heavy launch history tables.

### Step 2. Data Wrangling & Cleaning
- Extracted nested JSON data from API responses.
- Handled missing values (e.g., filled PayloadMass with mean, LandingPad with mode).
- Filtered dataset to only include Falcon 9 launches (excluded Falcon 1).
- Created a new Class column for landing success/failure.

### Step 3. Exploratory Data Analysis (EDA)
- **SQL Analysis:** Queried launch data to find patterns (success rates, payload mass, launch sites).
- **Visualizations:** Examined distributions, correlations, and trends in launch data.

### step 4. Feature Engineering
Extracted and transformed features like:
- BoosterVersion
- Orbit type
- LaunchSite
- PayloadMass
- Flights (number of previous flights)
- ReusedCount
- Geographic coordinates (Longitude, Latitude)

### step 5. Data Storage
- dataset_part_1.csv – Initial cleaned dataset
- dataset_part_2.csv – Dataset with Class label added
- dataset_part_3.csv - A cleaned and feature-engineered dataset ready for machine learning
- spacex_web_scraped.csv – Wikipedia scraped data
- my_data1.db – SQLite database with launch records

Final dataset saved for machine learning modeling.

## 📊 Process Breakdown (SpaceX_landing_Success_prediction.ipynb- ML operation)

### Step 1. Data Loading 
**Loads pre-processed datasets:** dataset_part_3.csv (features) and dataset_part_2.csv (target variable Class)

### Step 2. Data Preprocessing
**StandardScaler:** Normalizes features to have mean=0 and variance=1

**Train-Test Split:** 80-20 split with random state for reproducibility

### Step 3. Model Training & Selection
Four classification models are trained:
- Logistic Regression (with increased max_iter)
- K-Nearest Neighbors (K=5)
- Decision Tree Classifier (random_state=42)
- XGBoost Classifier (with eval_metric='logloss')

Cross-Validation Evaluation
Models are evaluated using 5-fold cross-validation:

- Logistic Regression: 83.5% ± 6.5%
- KNN: 66.7% ± 2.6%
- Decision Tree: 80.6% ± 11.4%
- XGBoost: 80.5% ± 9.6%

Hyperparameter Tuning (GridSearchCV)
- XGBoost is optimized with GridSearchCV:
- Parameters tested: n_estimators [50, 100, 200], max_depth [3, 5, 7], learning_rate [0.1, 0.01]
- Best parameters: {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 100}

**Best CV accuracy: 86.1%**

### Step 4. Model Evaluation
Best Model Performance (Tuned XGBoost)
Test Accuracy: 83.3%

**Confusion Matrix:**
[[ 3  3]  # 3 correct failures, 3 failures predicted as success
 [ 0 12]] # 0 successes predicted as failure, 12 correct successes
 
**Classification Report:**

- Failure class: 100% precision, 50% recall
- Success class: 80% precision, 100% recall

Model is better at predicting successes than failures

**ROC Curve Analysis**
AUC Score: 0.89 (good discriminatory power)
ROC curve shows strong performance above random chance line

### Step 5. Final Model Comparison
All models tested on test set:

- Decision Tree: 94.4% (best but may be overfitting)
- XGBoost: 83.3%
- Logistic Regression: 83.3%
- KNN: 66.7%

### Step 6. Model Persistence
The tuned XGBoost model and StandardScaler are saved using pickle:
- best_model.pkl: Trained XGBoost mode
- scaler.pkl: Fitted StandardScaler for future data preprocessing

## 🔧 Technologies & Libraries Used
- Python
- Pandas & NumPy (data manipulation)
- Requests (API calls)
- BeautifulSoup (web scraping)
- SQLite (database for SQL queries)
- Matplotlib & Seaborn (visualization)
- Jupyter Notebook (interactive analysis)
- 
## 👨‍💻 Beginner-Friendly Notes
This is a complete data pipeline project – from raw data to ML-ready dataset.

Great example of real-world data engineering: APIs, web scraping, cleaning, SQL, visualization.

The Class column is what we're trying to predict – everything else is a feature.

SQL queries help us ask business questions before modeling.

## Project Structure

```
📦 SpaceX-Falcon-9-Landing-Prediction
├── 📂 Configs
│   └── requirements.txt          # Project dependencies
│
├── 📂 datasets
│   ├── dataset_part_1.csv         # Raw dataset (initial data)
│   ├── dataset_part_2.csv         # Cleaned & processed dataset
│   ├── dataset_part_3.csv         # Final dataset used for modeling
│   ├── spacex_web_scraped.csv     # Data collected via web scraping
│   └── my_data1.db                # SQLite database file
│
├── 📂 examples
│   └── sample_request.json        # Sample input for API / testing
│
├── 📂 notebooks
│   ├── SpaceX Falcon 9 first stage Landing.ipynb
│   └── SpaceX_landing_Success_prediction.ipynb
│
├── 📂 src
│   └── app.py                     # Streamlit web application
│
├── .gitignore                     # Files & folders ignored by Git
├── Dockerfile                     # Docker configuration
└── README.md                      # Project documentation

```

