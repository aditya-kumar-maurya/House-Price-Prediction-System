# House Price Prediction System

A Flask + Machine Learning project inspired by the dashboard screenshot supplied with the request. It includes house-price prediction, interactive property management, CSV import, prediction history, analytics charts, model evaluation, PostgreSQL, and Docker Compose.

## Technology stack

- **Python** – application and model logic
- **NumPy / Pandas** – numerical operations and data preparation
- **Matplotlib / Seaborn** – analytics visualizations
- **Flask** – web application and routing
- **scikit-learn** – preprocessing and Random Forest regression
- **PostgreSQL** – persistent property and prediction-history storage
- **SQLAlchemy / psycopg** – database connection layer
- **Docker / Docker Compose** – repeatable app + database startup
- **HTML / CSS / JavaScript** – responsive dashboard UI

## Charts included in the ZIP

The ZIP includes five preview PNG charts under `static/charts/` generated with Matplotlib and Seaborn: price distribution, price vs area, average price by city, house type distribution, and average price by bedrooms. The running dashboard regenerates these charts from database records. The packaged previews use synthetic demo data.

## Features

- Dashboard KPIs and five analytics charts
- Predict estimated house price from city, area, bedrooms, bathrooms, house type, furnishing, parking, and construction year
- Show a simple lower/upper estimate range
- Property inventory: add, search, list and delete properties
- Prediction history: list and delete prediction records
- CSV import into PostgreSQL and model retraining
- Model page with MAE, RMSE and R²
- Dataset and technology overview
- Health endpoint at `/health`
- Synthetic demo data generated automatically when no training CSV exists

> **Data note:** By default, the app generates synthetic demo data. This makes the project runnable without finding a dataset, but predictions are not real market valuations. Import a representative dataset before using the model for serious analysis.

## Run with Docker (recommended)

Install Docker Desktop and ensure Docker is running. In this folder, open a terminal and run:

```bash
docker compose up --build
```

Open `http://localhost:5000`.

Stop the app with:

```bash
docker compose down
```

Stop and delete the persisted PostgreSQL volume (this permanently deletes its database data):

```bash
docker compose down -v
```

Docker Compose starts PostgreSQL first, waits for its health check, then starts the Flask/Gunicorn app. The demo credentials are in `docker-compose.yml`; change them before deployment.

## Run locally without Docker

1. Install Python 3.11 or newer and PostgreSQL.
2. Create a PostgreSQL database and user, or update the connection string to match your local setup.
3. Open a terminal in this project folder.
4. Create and activate a virtual environment:

   **Windows PowerShell**
   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   **macOS/Linux**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

5. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
6. Set the database URL.

   **Windows PowerShell**
   ```powershell
   $env:DATABASE_URL="postgresql+psycopg://houseuser:housepass@localhost:5432/houseprices"
   ```

   **macOS/Linux**
   ```bash
   export DATABASE_URL="postgresql+psycopg://houseuser:housepass@localhost:5432/houseprices"
   ```
7. Start the app:
   ```bash
   python app.py
   ```
8. Visit `http://127.0.0.1:5000`.

The default database URL in `app.py` is SQLite as a convenience fallback for local experimentation. For the requested PostgreSQL setup, use the environment variable above or Docker Compose.

## CSV dataset format

Upload CSV from the Dataset page. Column names must match exactly:

```csv
city,area_sqft,bedrooms,bathrooms,house_type,furnishing_status,parking,year_built,price_lakh
Hyderabad,1200,3,2,Apartment,Semi-Furnished,1,2018,64.5
Bangalore,1500,3,3,Independent House,Furnished,2,2021,76.2
Mumbai,1100,2,2,Apartment,Unfurnished,1,2016,82.7
```

Prices are in Indian lakh (₹ lakh). Keep units and category names consistent across your dataset.

## Pages

| URL | Purpose |
|---|---|
| `/` | Dashboard |
| `/predict` | House price prediction |
| `/analysis` | Charts and feature correlation table |
| `/houses` | Property CRUD |
| `/dataset` | CSV import and data preview |
| `/model` | Model metrics and retraining |
| `/history` | Prediction history |
| `/about` | Project and technology overview |
| `/health` | App/database health check |

## Model workflow

1. Read CSV data if `data/house_prices.csv` exists; otherwise generate synthetic sample data.
2. Split features and target (`price_lakh`) into training and test sets.
3. Impute missing numeric/categorical values.
4. Standardize numeric columns and one-hot encode categorical columns.
5. Train a `RandomForestRegressor`.
6. Evaluate with MAE, RMSE and R².
7. Save the pipeline to `house_price_model.joblib`.

## Project structure

```text
house_price_prediction_system/
├── app.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── README.md
├── data/
│   └── README.txt
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── predict.html
│   ├── analysis.html
│   ├── houses.html
│   ├── history.html
│   ├── dataset.html
│   ├── model.html
│   ├── about.html
│   └── error.html
└── static/
    ├── css/style.css
    ├── js/app.js
    └── charts/  (generated at runtime)
```

## Troubleshooting

- **Database connection error:** verify PostgreSQL is running and `DATABASE_URL` has the correct user, password, host, port and database.
- **Port 5000 already in use:** set `PORT=5001` and map the Docker port accordingly, or stop the other process.
- **Model retraining fails after CSV import:** verify every required column is present, numeric fields contain numeric values, and `price_lakh` has valid values.
- **Charts are missing:** refresh the dashboard after startup and check that `static/charts/` is writable.

## Academic project disclaimer

This is a learning project. Synthetic data is only for demonstration; model scores on synthetic data do not demonstrate real-world accuracy. Always validate with representative data before drawing conclusions.
