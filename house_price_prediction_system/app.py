import os
from dotenv import load_dotenv
from datetime import datetime
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from flask import (
    Flask, render_template, request, redirect,
    url_for, flash, send_file
)

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import joblib


# --------------------------------------------------
# 1. LOAD ENVIRONMENT VARIABLES
# --------------------------------------------------

load_dotenv()

app = Flask(__name__)

app.secret_key = os.getenv(
    "FLASK_SECRET_KEY",
    "change-this-development-secret"
)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///house_prices.db"
).strip()

# Convert PostgreSQL URLs to use Psycopg 3.
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = "postgresql+psycopg://" + DATABASE_URL[len("postgres://"):]

elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = "postgresql+psycopg://" + DATABASE_URL[len("postgresql://"):]


MODEL_PATH = "house_price_model.joblib"
DATA_PATH = "data/house_prices.csv"


# --------------------------------------------------
# 2. MACHINE LEARNING FEATURES
# --------------------------------------------------

NUMERIC_FEATURES = [
    "area_sqft",
    "bedrooms",
    "bathrooms",
    "parking",
    "year_built"
]

CATEGORICAL_FEATURES = [
    "city",
    "house_type",
    "furnishing_status"
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "price_lakh"

os.makedirs("data", exist_ok=True)
os.makedirs("static/charts", exist_ok=True)


# --------------------------------------------------
# 3. DATABASE CONNECTION
# --------------------------------------------------

def get_engine():
    engine_options = {
        "pool_pre_ping": True,
        "future": True
    }

    # Supabase transaction pooler compatibility.
    if DATABASE_URL.startswith("postgresql+psycopg://"):
        engine_options["connect_args"] = {
            "prepare_threshold": None
        }

    return create_engine(DATABASE_URL, **engine_options)


engine = get_engine()


# --------------------------------------------------
# 4. CREATE SAMPLE DATA
# --------------------------------------------------

def create_sample_data(n=1000, seed=42):
    """Generate sample house-price data for model training."""

    rng = np.random.default_rng(seed)

    cities = np.array([
        "Hyderabad", "Bangalore", "Mumbai", "Delhi",
        "Pune", "Chennai", "Kolkata", "Noida",
        "Gurgaon", "Ahmedabad"
    ])

    city_base = {
        "Hyderabad": 42,
        "Bangalore": 48,
        "Mumbai": 65,
        "Delhi": 45,
        "Pune": 36,
        "Chennai": 34,
        "Kolkata": 27,
        "Noida": 32,
        "Gurgaon": 51,
        "Ahmedabad": 25
    }

    city = rng.choice(cities, n)
    area = rng.integers(450, 3201, n)

    bedrooms = np.clip(
        np.round(area / 520 + rng.normal(0, 0.8, n)),
        1, 6
    ).astype(int)

    bathrooms = np.clip(
        bedrooms + rng.integers(-1, 2, n),
        1, 5
    )

    parking = rng.integers(0, 4, n)
    year_built = rng.integers(1985, 2027, n)

    house_type = rng.choice(
        ["Apartment", "Independent House", "Villa", "Builder Floor"],
        n,
        p=[0.46, 0.25, 0.16, 0.13]
    )

    furnishing = rng.choice(
        ["Unfurnished", "Semi-Furnished", "Furnished"],
        n,
        p=[0.30, 0.43, 0.27]
    )

    furnishing_bonus = (
        pd.Series(furnishing)
        .map({
            "Unfurnished": 0,
            "Semi-Furnished": 3.5,
            "Furnished": 7
        })
        .to_numpy()
    )

    type_bonus = (
        pd.Series(house_type)
        .map({
            "Apartment": 0,
            "Independent House": 7,
            "Villa": 15,
            "Builder Floor": 3
        })
        .to_numpy()
    )

    age_penalty = np.maximum(0, 2026 - year_built) * 0.10

    price = (
        area * 0.018
        + bedrooms * 2.3
        + bathrooms * 1.4
        + parking * 1.1
        + np.array([city_base[c] for c in city])
        + furnishing_bonus
        + type_bonus
        - age_penalty
        + rng.normal(0, 7, n)
    )

    price = np.maximum(price, 12).round(2)

    return pd.DataFrame({
        "city": city,
        "area_sqft": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "house_type": house_type,
        "furnishing_status": furnishing,
        "parking": parking,
        "year_built": year_built,
        "price_lakh": price
    })


# --------------------------------------------------
# 5. INITIALIZE DATABASE TABLES
# --------------------------------------------------

def initialize_database():

    is_postgres = engine.dialect.name == "postgresql"

    if is_postgres:
        properties_table_sql = """
        CREATE TABLE IF NOT EXISTS properties (
            id INTEGER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            city VARCHAR(100) NOT NULL,
            area_sqft INTEGER NOT NULL,
            bedrooms INTEGER NOT NULL,
            bathrooms INTEGER NOT NULL,
            house_type VARCHAR(100) NOT NULL,
            furnishing_status VARCHAR(100) NOT NULL,
            parking INTEGER NOT NULL DEFAULT 0,
            year_built INTEGER NOT NULL,
            price_lakh DOUBLE PRECISION NOT NULL,
            status VARCHAR(30) NOT NULL DEFAULT 'Available',
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """

        history_table_sql = """
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            city VARCHAR(100) NOT NULL,
            area_sqft INTEGER NOT NULL,
            bedrooms INTEGER NOT NULL,
            bathrooms INTEGER NOT NULL,
            house_type VARCHAR(100) NOT NULL,
            furnishing_status VARCHAR(100) NOT NULL,
            parking INTEGER NOT NULL,
            year_built INTEGER NOT NULL,
            predicted_price DOUBLE PRECISION NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """

    else:
        properties_table_sql = """
        CREATE TABLE IF NOT EXISTS properties (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            city TEXT NOT NULL,
            area_sqft INTEGER NOT NULL,
            bedrooms INTEGER NOT NULL,
            bathrooms INTEGER NOT NULL,
            house_type TEXT NOT NULL,
            furnishing_status TEXT NOT NULL,
            parking INTEGER NOT NULL DEFAULT 0,
            year_built INTEGER NOT NULL,
            price_lakh REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'Available',
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """

        history_table_sql = """
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            city TEXT NOT NULL,
            area_sqft INTEGER NOT NULL,
            bedrooms INTEGER NOT NULL,
            bathrooms INTEGER NOT NULL,
            house_type TEXT NOT NULL,
            furnishing_status TEXT NOT NULL,
            parking INTEGER NOT NULL,
            year_built INTEGER NOT NULL,
            predicted_price REAL NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """

    with engine.begin() as conn:

        conn.execute(text(properties_table_sql))
        conn.execute(text(history_table_sql))

        count = conn.execute(
            text("SELECT COUNT(*) FROM properties")
        ).scalar_one()

        # Add initial sample properties when the table is empty.
        if count == 0:

            sample = create_sample_data(120)

            for _, row in sample.iterrows():

                conn.execute(
                    text("""
                        INSERT INTO properties (
                            city, area_sqft, bedrooms, bathrooms,
                            house_type, furnishing_status, parking,
                            year_built, price_lakh, status
                        )
                        VALUES (
                            :city, :area_sqft, :bedrooms, :bathrooms,
                            :house_type, :furnishing_status, :parking,
                            :year_built, :price_lakh, 'Available'
                        )
                    """),
                    {
                        key: (
                            int(value)
                            if key in [
                                "area_sqft", "bedrooms", "bathrooms",
                                "parking", "year_built"
                            ]
                            else float(value)
                            if key == "price_lakh"
                            else str(value)
                        )
                        for key, value in row.to_dict().items()
                    }
                )


# --------------------------------------------------
# 6. GET TRAINING DATA
# --------------------------------------------------

def get_training_data():

    if os.path.exists(DATA_PATH):

        df = pd.read_csv(DATA_PATH)

        missing = [
            col for col in FEATURES + [TARGET]
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                "CSV is missing required columns: "
                + ", ".join(missing)
            )

        return df[FEATURES + [TARGET]].dropna(subset=[TARGET])

    return create_sample_data(1000)


# --------------------------------------------------
# 7. TRAIN MACHINE LEARNING MODEL
# --------------------------------------------------

def train_model(force=False):

    if os.path.exists(MODEL_PATH) and not force:
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            pass

    df = get_training_data()

    X = df[FEATURES]
    y = df[TARGET]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=180,
                random_state=42,
                min_samples_leaf=2
            )
        )
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    model_metrics = {
        "mae": round(
            float(mean_absolute_error(y_test, predictions)), 2
        ),
        "rmse": round(
            float(np.sqrt(mean_squared_error(y_test, predictions))), 2
        ),
        "r2": round(
            float(r2_score(y_test, predictions)), 3
        )
    }

    result = {
        "model": model,
        "metrics": model_metrics,
        "features": FEATURES
    }

    joblib.dump(result, MODEL_PATH)

    return result


# --------------------------------------------------
# 8. DATABASE QUERY HELPER
# --------------------------------------------------

def query_df(sql, params=None):

    with engine.connect() as conn:
        return pd.read_sql(
            text(sql),
            conn,
            params=params or {}
        )


# --------------------------------------------------
# 9. GENERATE DARK BLUE ANALYTICS CHARTS
# --------------------------------------------------

def make_charts():

    df = query_df("""
        SELECT city, area_sqft, bedrooms, bathrooms,
               house_type, furnishing_status,
               year_built, price_lakh
        FROM properties
    """)

    if df.empty:
        return

    # Dark blue theme similar to Student Performance Analysis
    sns.set_theme(
        style="darkgrid",
        rc={
            "figure.facecolor": "#0b1730",
            "axes.facecolor": "#101d35",
            "axes.edgecolor": "#35527a",
            "axes.labelcolor": "#dce8ff",
            "text.color": "#e6eefb",
            "xtick.color": "#c5d5ee",
            "ytick.color": "#c5d5ee",
            "grid.color": "#2a4264",
            "grid.alpha": 0.45,
            "axes.titlecolor": "#e6eefb",
            "axes.titleweight": "bold",
            "savefig.facecolor": "#0b1730",
            "savefig.edgecolor": "#0b1730"
        }
    )

    def style_chart(fig, ax, title, xlabel=None, ylabel=None):

        fig.patch.set_facecolor("#0b1730")
        ax.set_facecolor("#101d35")

        ax.set_title(
            title,
            color="#e6eefb",
            fontsize=11,
            fontweight="bold",
            pad=12
        )

        if xlabel:
            ax.set_xlabel(xlabel, color="#c5d5ee")

        if ylabel:
            ax.set_ylabel(ylabel, color="#c5d5ee")

        ax.tick_params(
            axis="both",
            colors="#c5d5ee",
            labelsize=8
        )

        ax.grid(
            True,
            color="#2a4264",
            alpha=0.45,
            linewidth=0.7
        )

        for spine in ax.spines.values():
            spine.set_color("#35527a")

    def save_chart(fig, filename):

        fig.tight_layout()

        fig.savefig(
            filename,
            dpi=150,
            facecolor="#0b1730",
            edgecolor="none",
            bbox_inches="tight"
        )

        plt.close(fig)

    # 1. Price Distribution
    fig, ax = plt.subplots(figsize=(5, 3))

    sns.histplot(
        df["price_lakh"],
        bins=18,
        kde=True,
        ax=ax,
        color="#20c4f4",
        edgecolor="#0b1730",
        alpha=0.9
    )

    style_chart(
        fig,
        ax,
        "Price Distribution",
        "Price (₹ lakh)",
        "Count"
    )

    save_chart(
        fig,
        "static/charts/price_distribution1.png"
    )

    # 2. Price vs Area
    fig, ax = plt.subplots(figsize=(5, 3))

    sns.scatterplot(
        data=df,
        x="area_sqft",
        y="price_lakh",
        hue="city",
        palette="bright",
        legend=False,
        alpha=0.85,
        s=30,
        ax=ax
    )

    sns.regplot(
        data=df,
        x="area_sqft",
        y="price_lakh",
        scatter=False,
        ax=ax,
        color="#ff8ac7",
        line_kws={"linewidth": 2}
    )

    style_chart(
        fig,
        ax,
        "Price vs Area",
        "Area (sq ft)",
        "Price (₹ lakh)"
    )

    save_chart(
        fig,
        "static/charts/price_vs_area.png"
    )

    # 3. Average Price by City
    by_city = (
        df.groupby("city")["price_lakh"]
        .mean()
        .sort_values(ascending=False)
        .head(5)
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(5, 3))

    by_city.plot(
        kind="barh",
        ax=ax,
        color=[
            "#20c4f4",
            "#38d9c5",
            "#9b8cff",
            "#ff8ac7",
            "#ffd166"
        ][:len(by_city)],
        edgecolor="none"
    )

    style_chart(
        fig,
        ax,
        "Average Price by City",
        "Price (₹ lakh)"
    )

    save_chart(
        fig,
        "static/charts/city_prices1.png"
    )

    # 4. House Type Distribution
    fig, ax = plt.subplots(figsize=(5, 3))

    type_colors = [
        "#20c4f4",
        "#38d9c5",
        "#ff8ac7",
        "#ffd166"
    ]

    df["house_type"].value_counts().plot(
        kind="pie",
        autopct="%1.0f%%",
        ax=ax,
        startangle=90,
        colors=type_colors,
        textprops={
            "color": "#ffffff",
            "fontsize": 8
        },
        wedgeprops={
            "edgecolor": "#0b1730",
            "linewidth": 2
        }
    )

    ax.set_facecolor("#101d35")
    fig.patch.set_facecolor("#0b1730")

    ax.set_title(
        "House Type Distribution",
        color="#e6eefb",
        fontsize=11,
        fontweight="bold",
        pad=12
    )

    ax.set_ylabel("")

    save_chart(
        fig,
        "static/charts/house_types1.png"
    )

    # 5. Average Price by Bedrooms
    fig, ax = plt.subplots(figsize=(5, 3))

    sns.barplot(
        data=df,
        x="bedrooms",
        y="price_lakh",
        estimator=np.mean,
        errorbar=None,
        ax=ax,
        color="#20c4f4",
        edgecolor="none"
    )

    style_chart(
        fig,
        ax,
        "Average Price by Bedrooms",
        "Bedrooms",
        "Average Price (₹ lakh)"
    )

    save_chart(
        fig,
        "static/charts/bedrooms1.png"
    )

# --------------------------------------------------
# 10. DASHBOARD METRICS
# --------------------------------------------------

def metrics():

    data = query_df("""
        SELECT
            COUNT(*) AS n,
            AVG(price_lakh) AS avg_price,
            AVG(area_sqft) AS avg_area
        FROM properties
    """)

    if not data.empty:
        return data.iloc[0].to_dict()

    return {
        "n": 0,
        "avg_price": 0,
        "avg_area": 0
    }


# --------------------------------------------------
# 11. HOME PAGE
# --------------------------------------------------

@app.route("/")
def home():

    try:

        stats = metrics()

        make_charts()

        recent = query_df("""
            SELECT *
            FROM prediction_history
            ORDER BY id DESC
            LIMIT 5
        """)

        model_data = train_model()

        return render_template(
            "index.html",
            stats=stats,
            recent=recent.to_dict("records"),
            model_metrics=model_data["metrics"]
        )

    except Exception as exc:

        return render_template(
            "error.html",
            error=str(exc)
        ), 500


# --------------------------------------------------
# 12. HOUSE PRICE PREDICTION
# --------------------------------------------------

@app.route("/predict", methods=["GET", "POST"])
def predict():

    result = None

    form = {
        "city": "Hyderabad",
        "area_sqft": "1200",
        "bedrooms": "3",
        "bathrooms": "2",
        "house_type": "Apartment",
        "furnishing_status": "Semi-Furnished",
        "parking": "1",
        "year_built": "2018"
    }

    if request.method == "POST":

        try:

            form = {
                key: request.form.get(key, "")
                for key in [
                    "city", "area_sqft", "bedrooms", "bathrooms",
                    "house_type", "furnishing_status",
                    "parking", "year_built"
                ]
            }

            row = {
                "city": form["city"],
                "area_sqft": int(form["area_sqft"]),
                "bedrooms": int(form["bedrooms"]),
                "bathrooms": int(form["bathrooms"]),
                "house_type": form["house_type"],
                "furnishing_status": form["furnishing_status"],
                "parking": int(form["parking"]),
                "year_built": int(form["year_built"])
            }

            if not 100 <= row["area_sqft"] <= 20000:
                raise ValueError(
                    "Area must be between 100 and 20,000 sq ft."
                )

            model_data = train_model()

            estimate = max(
                0,
                float(
                    model_data["model"].predict(
                        pd.DataFrame([row])[FEATURES]
                    )[0]
                )
            )

            low = estimate * 0.88
            high = estimate * 1.12

            with engine.begin() as conn:

                conn.execute(
                    text("""
                        INSERT INTO prediction_history (
                            city, area_sqft, bedrooms, bathrooms,
                            house_type, furnishing_status, parking,
                            year_built, predicted_price
                        )
                        VALUES (
                            :city, :area_sqft, :bedrooms, :bathrooms,
                            :house_type, :furnishing_status, :parking,
                            :year_built, :predicted_price
                        )
                    """),
                    {
                        **row,
                        "predicted_price": estimate
                    }
                )

            result = {
                "price": round(estimate, 2),
                "low": round(low, 2),
                "high": round(high, 2)
            }

        except Exception as exc:
            flash(f"Could not predict price: {exc}", "error")

    return render_template(
        "predict.html",
        form=form,
        result=result
    )


# --------------------------------------------------
# 13. ANALYSIS PAGE
# --------------------------------------------------

@app.route("/analysis")
def analysis():

    make_charts()

    data = query_df("""
        SELECT city, house_type, furnishing_status,
               area_sqft, bedrooms, bathrooms,
               parking, year_built, price_lakh
        FROM properties
    """)

    corr_cols = [
        "area_sqft", "bedrooms", "bathrooms",
        "parking", "year_built", "price_lakh"
    ]

    corr = (
        data[corr_cols].corr(numeric_only=True).round(2).to_dict()
        if not data.empty else {}
    )

    return render_template(
        "analysis.html",
        corr=corr,
        columns=corr_cols
    )


# --------------------------------------------------
# 14. HOUSE MANAGEMENT
# --------------------------------------------------

@app.route("/houses")
def houses():

    q = request.args.get("q", "").strip()

    if q:

        sql = """
            SELECT *
            FROM properties
            WHERE LOWER(city) LIKE LOWER(:q)
               OR LOWER(house_type) LIKE LOWER(:q)
            ORDER BY id DESC
        """

        data = query_df(sql, {"q": f"%{q}%"})

    else:

        data = query_df(
            "SELECT * FROM properties ORDER BY id DESC"
        )

    return render_template(
        "houses.html",
        houses=data.to_dict("records"),
        q=q
    )


@app.route("/houses/add", methods=["POST"])
def add_house():

    try:

        row = read_house_form(request.form)

        with engine.begin() as conn:

            conn.execute(
                text("""
                    INSERT INTO properties (
                        city, area_sqft, bedrooms, bathrooms,
                        house_type, furnishing_status, parking,
                        year_built, price_lakh, status
                    )
                    VALUES (
                        :city, :area_sqft, :bedrooms, :bathrooms,
                        :house_type, :furnishing_status, :parking,
                        :year_built, :price_lakh, :status
                    )
                """),
                row
            )

        flash("Property added successfully.", "success")

    except Exception as exc:
        flash(f"Could not add property: {exc}", "error")

    return redirect(url_for("houses"))


def read_house_form(form):

    row = {
        "city": form["city"].strip(),
        "area_sqft": int(form["area_sqft"]),
        "bedrooms": int(form["bedrooms"]),
        "bathrooms": int(form["bathrooms"]),
        "house_type": form["house_type"],
        "furnishing_status": form["furnishing_status"],
        "parking": int(form["parking"]),
        "year_built": int(form["year_built"]),
        "price_lakh": float(form["price_lakh"]),
        "status": form.get("status", "Available")
    }

    if (
        not row["city"]
        or row["area_sqft"] < 100
        or row["price_lakh"] <= 0
    ):
        raise ValueError(
            "Enter a city, valid area, and positive price."
        )

    return row


@app.route("/houses/delete/<int:house_id>", methods=["POST"])
def delete_house(house_id):

    with engine.begin() as conn:

        conn.execute(
            text("DELETE FROM properties WHERE id=:id"),
            {"id": house_id}
        )

    flash("Property deleted.", "success")

    return redirect(url_for("houses"))


# --------------------------------------------------
# 15. PREDICTION HISTORY
# --------------------------------------------------

@app.route("/history")
def history():

    data = query_df("""
        SELECT *
        FROM prediction_history
        ORDER BY id DESC
        LIMIT 200
    """)

    return render_template(
        "history.html",
        history=data.to_dict("records")
    )


@app.route(
    "/history/delete/<int:prediction_id>",
    methods=["POST"]
)
def delete_prediction(prediction_id):

    with engine.begin() as conn:

        conn.execute(
            text("DELETE FROM prediction_history WHERE id=:id"),
            {"id": prediction_id}
        )

    flash("Prediction record deleted.", "success")

    return redirect(url_for("history"))


# --------------------------------------------------
# 16. DATASET PAGE AND CSV IMPORT
# --------------------------------------------------

@app.route("/dataset")
def dataset():

    data = query_df("""
        SELECT *
        FROM properties
        ORDER BY id DESC
        LIMIT 250
    """)

    return render_template(
        "dataset.html",
        rows=data.to_dict("records")
    )


@app.route("/dataset/upload", methods=["POST"])
def upload_dataset():

    file = request.files.get("csv_file")

    if not file or not file.filename.lower().endswith(".csv"):

        flash("Please choose a CSV file.", "error")

        return redirect(url_for("dataset"))

    try:

        df = pd.read_csv(file)

        required = FEATURES + [TARGET]

        missing = [
            col for col in required
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                "CSV columns missing: " + ", ".join(missing)
            )

        df = df[required].dropna(subset=[TARGET])

        with engine.begin() as conn:

            for _, row in df.iterrows():

                conn.execute(
                    text("""
                        INSERT INTO properties (
                            city, area_sqft, bedrooms, bathrooms,
                            house_type, furnishing_status, parking,
                            year_built, price_lakh, status
                        )
                        VALUES (
                            :city, :area_sqft, :bedrooms, :bathrooms,
                            :house_type, :furnishing_status, :parking,
                            :year_built, :price_lakh, 'Available'
                        )
                    """),
                    {
                        key: (
                            int(row[key])
                            if key in [
                                "area_sqft", "bedrooms",
                                "bathrooms", "parking", "year_built"
                            ]
                            else float(row[key])
                            if key == TARGET
                            else str(row[key])
                        )
                        for key in required
                    }
                )

        train_model(force=True)

        flash(
            f"Imported {len(df)} rows and retrained the model.",
            "success"
        )

    except Exception as exc:

        flash(f"CSV import failed: {exc}", "error")

    return redirect(url_for("dataset"))


# --------------------------------------------------
# 17. MODEL PAGE
# --------------------------------------------------

@app.route("/model", methods=["GET", "POST"])
def model_page():

    if request.method == "POST":

        try:

            train_model(force=True)

            flash(
                "Model retrained successfully.",
                "success"
            )

        except Exception as exc:

            flash(
                f"Model training failed: {exc}",
                "error"
            )

    data = train_model()

    return render_template(
        "model.html",
        metrics=data["metrics"],
        features=FEATURES
    )


# --------------------------------------------------
# 18. ABOUT PAGE
# --------------------------------------------------

@app.route("/about")
def about():
    return render_template("about.html")


# --------------------------------------------------
# 19. DATABASE HEALTH CHECK
# --------------------------------------------------

@app.route("/health")
def health():

    try:

        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        return {
            "status": "ok",
            "database": engine.dialect.name
        }

    except Exception as exc:

        app.logger.exception("Database health check failed")

        return {
            "status": "error",
            "message": str(exc)
        }, 500


# --------------------------------------------------
# 20. STARTUP
# --------------------------------------------------

with app.app_context():

    try:

        initialize_database()
        train_model()

        print("Database:", engine.dialect.name)
        print("Database initialization completed.")

    except Exception as startup_error:

        print("Startup setup warning:", startup_error)


# --------------------------------------------------
# 21. RUN FLASK APPLICATION
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "0") == "1"
    )

