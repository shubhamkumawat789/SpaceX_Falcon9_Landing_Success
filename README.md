
##SpaceX Falcon 9 First Stage Landing Prediction
###📌 Project Overview
This project aims to predict whether the Falcon 9 rocket's first stage will land successfully after launch. Using data from SpaceX's API and Wikipedia, we analyze historical launch data to build a classification model that can determine landing outcomes.

###🎯 Objective
To build a machine learning model that can predict if the Falcon 9's first stage will land successfully based on features like:

FEATURE DESCRIPTIONS

+------------------+--------------------------------------------------------------+
| Feature Name | Description |
+------------------+--------------------------------------------------------------+
| LaunchSite | Location where the rocket was launched |
| | (e.g., CCAFS SLC 40, KSC LC 39A, VAFB SLC 4E) |
+------------------+--------------------------------------------------------------+
| Orbit | Destination orbit of the payload |
| | (e.g., LEO, GTO, ISS, PO, SSO) |
+------------------+--------------------------------------------------------------+
| BoosterVersion | Specific model or variant of the Falcon 9 booster |
+------------------+--------------------------------------------------------------+
| ReuseHistory | Indicates whether the booster was reused |
| | from previous flights |
+------------------+--------------------------------------------------------------+
| FlightNumber | Sequential number of the launch mission |
+------------------+--------------------------------------------------------------+
| PayloadMass | Mass of the payload carried by the rocket (in kg) |
+------------------+--------------------------------------------------------------+
| Flights | Number of times the booster has flown before |
| | the current mission |
+------------------+--------------------------------------------------------------+
| Block | Booster block version (e.g., Block 1, Block 5) |
+------------------+--------------------------------------------------------------+
| GridFins | Whether grid fins were used for steering |
| | during descent (1 = Yes, 0 = No) |
+------------------+--------------------------------------------------------------+
| Legs | Whether landing legs were deployed |
| | (1 = Yes, 0 = No) |
+------------------+--------------------------------------------------------------+

🔑 Key Points
Target Variable: Class (1 = successful landing, 0 = failed landing)

Data Sources:

SpaceX REST API (via requests)

Wikipedia web scraping (using BeautifulSoup)

Tools Used: Python, Jupyter Notebook, Pandas, SQLite, Matplotlib/Seaborn

Problem Type: Binary classification

🔧 Technologies & Libraries Used
text
- Python
- Pandas & NumPy (data manipulation)
- Requests (API calls)
- BeautifulSoup (web scraping)
- SQLite (database for SQL queries)
- Matplotlib & Seaborn (visualization)
- Jupyter Notebook (interactive analysis)
📊 Process Breakdown
1. Data Collection
From SpaceX API: Launch details, rocket info, payload data, core data, launchpad info.

From Wikipedia: Scraped Falcon 9 and Falcon Heavy launch history tables.

2. Data Wrangling & Cleaning
Extracted nested JSON data from API responses.

Handled missing values (e.g., filled PayloadMass with mean, LandingPad with mode).

Filtered dataset to only include Falcon 9 launches (excluded Falcon 1).

Created a new Class column for landing success/failure.

3. Exploratory Data Analysis (EDA)
SQL Analysis: Queried launch data to find patterns (success rates, payload mass, launch sites).

Visualizations: Examined distributions, correlations, and trends in launch data.

4. Feature Engineering
Extracted and transformed features like:

BoosterVersion

Orbit type

LaunchSite

PayloadMass

Flights (number of previous flights)

ReusedCount

Geographic coordinates (Longitude, Latitude)

5. Data Storage
Saved cleaned data to CSV (dataset_part_1.csv, dataset_part_2.csv).

Loaded data into SQLite database for querying.

📈 In-Depth Analysis Steps
✅ Step 1: API Data Extraction
Created helper functions to fetch:

Booster version

Launch site details (name, lat/long)

Payload data (mass, orbit)

Core data (reuse count, landing outcome)

✅ Step 2: Data Filtering & Transformation
Removed launches with multiple payloads/cores.

Converted date columns to proper datetime format.

Filtered to dates before November 13, 2020.

✅ Step 3: Handling Missing Data
PayloadMass: Replaced NaN with column mean.

LandingPad: Filled NaN with mode (most frequent value).

✅ Step 4: Label Creation
Landing outcomes categorized as:

Good outcomes: True ASDS, True RTLS, True Ocean

Bad outcomes: None None, False Ocean, False RTLS, False ASDS, None ASDS

Created binary Class column (1 for success, 0 for failure).

✅ Step 5: SQL-Based EDA
Performed queries to find:

Unique launch sites

Total payload mass for NASA

First successful ground pad landing

Success/failure counts

And more...

✅ Step 6: Data Export
Final dataset saved for machine learning modeling.

🚀 Next Steps (Implied)
This notebook sets up the data for:

Feature selection for the ML model

Train/test split

Model training (likely classification algorithms)

Model evaluation and prediction

Deployment of the landing predictor

📁 Output Files
dataset_part_1.csv – Initial cleaned dataset

dataset_part_2.csv – Dataset with Class label added

spacex_web_scraped.csv – Wikipedia scraped data

my_data1.db – SQLite database with launch records

👨‍💻 Beginner-Friendly Notes
This is a complete data pipeline project – from raw data to ML-ready dataset.

Great example of real-world data engineering: APIs, web scraping, cleaning, SQL, visualization.

The Class column is what we're trying to predict – everything else is a feature.

SQL queries help us ask business questions before modeling.


## 3) Project layout

```
spacex-landing-starter/
├─ configs/config.yaml
├─ data/                       # your CSV goes here (gitignored)
├─ examples/sample_request.json
├─ src/spacex_landing/
│  ├─ data.py                  # load CSV, basic checks
│  ├─ features.py              # feature preprocessing
│  ├─ train.py                 # train & save model
│  ├─ inference.py             # load & predict
│  └─ serving/
│     ├─ schemas.py            # pydantic request/response
│     └─ api.py                # FastAPI app
├─ tests/test_smoke.py         # very simple tests
└─ README.md
```

