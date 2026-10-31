import os
import tempfile
from datetime import datetime
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
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

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "change-this-development-secret")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///house_prices.db").strip()
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = "postgresql+psycopg://" + DATABASE_URL[len("postgres://"):]
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = "postgresql+psycopg://" + DATABASE_URL[len("postgresql://"):]
MODEL_PATH = os.path.join(tempfile.gettempdir(), "house_price_model.joblib")
DATA_PATH = "data/house_prices.csv"

NUMERIC_FEATURES = ["area_sqft", "bedrooms", "bathrooms", "parking", "year_built"]
CATEGORICAL_FEATURES = ["city", "house_type", "furnishing_status"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "price_lakh"

os.makedirs("data", exist_ok=True)

def get_engine():
    return create_engine(DATABASE_URL, pool_pre_ping=True, future=True)

engine = get_engine()

def create_sample_data(n=1000, seed=42):
    """Create reproducible demo data so the app works without downloading a dataset."""
    rng = np.random.default_rng(seed)
    cities = np.array(["Hyderabad", "Bangalore", "Mumbai", "Delhi", "Pune", "Chennai", "Kolkata", "Noida", "Gurgaon", "Ahmedabad"])
    city_base = {"Hyderabad": 42, "Bangalore": 48, "Mumbai": 65, "Delhi": 45, "Pune": 36,
                 "Chennai": 34, "Kolkata": 27, "Noida": 32, "Gurgaon": 51, "Ahmedabad": 25}
    city = rng.choice(cities, n)
    area = rng.integers(450, 3201, n)
    bedrooms = np.clip(np.round(area / 520 + rng.normal(0, 0.8, n)), 1, 6).astype(int)
    bathrooms = np.clip(bedrooms + rng.integers(-1, 2, n), 1, 5)
    parking = rng.integers(0, 4, n)
    year_built = rng.integers(1985, 2027, n)
    house_type = rng.choice(["Apartment", "Independent House", "Villa", "Builder Floor"], n, p=[.46, .25, .16, .13])
    furnishing = rng.choice(["Unfurnished", "Semi-Furnished", "Furnished"], n, p=[.30, .43, .27])
    furnishing_bonus = pd.Series(furnishing).map({"Unfurnished": 0, "Semi-Furnished": 3.5, "Furnished": 7}).to_numpy()
    type_bonus = pd.Series(house_type).map({"Apartment": 0, "Independent House": 7, "Villa": 15, "Builder Floor": 3}).to_numpy()
    age_penalty = np.maximum(0, 2026 - year_built) * .10
    price = (area * .018 + bedrooms * 2.3 + bathrooms * 1.4 + parking * 1.1 +
             np.array([city_base[c] for c in city]) + furnishing_bonus + type_bonus - age_penalty +
             rng.normal(0, 7, n))
    price = np.maximum(price, 12).round(2)
    df = pd.DataFrame({"city": city, "area_sqft": area, "bedrooms": bedrooms,
                       "bathrooms": bathrooms, "house_type": house_type,
                       "furnishing_status": furnishing, "parking": parking,
                       "year_built": year_built, "price_lakh": price})
    return df

def initialize_database():
    with engine.begin() as conn:
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS properties (
            id SERIAL PRIMARY KEY,
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
        """) if engine.dialect.name == "postgresql" else text("""
        CREATE TABLE IF NOT EXISTS properties (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            city TEXT NOT NULL, area_sqft INTEGER NOT NULL, bedrooms INTEGER NOT NULL,
            bathrooms INTEGER NOT NULL, house_type TEXT NOT NULL, furnishing_status TEXT NOT NULL,
            parking INTEGER NOT NULL DEFAULT 0, year_built INTEGER NOT NULL,
            price_lakh REAL NOT NULL, status TEXT NOT NULL DEFAULT 'Available',
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """))
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id %s,
            city VARCHAR(100) NOT NULL, area_sqft INTEGER NOT NULL, bedrooms INTEGER NOT NULL,
            bathrooms INTEGER NOT NULL, house_type VARCHAR(100) NOT NULL,
            furnishing_status VARCHAR(100) NOT NULL, parking INTEGER NOT NULL,
            year_built INTEGER NOT NULL, predicted_price DOUBLE PRECISION NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """ % ("INTEGER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY" if engine.dialect.name == "postgresql" else "INTEGER PRIMARY KEY AUTOINCREMENT")))
        count = conn.execute(text("SELECT COUNT(*) FROM properties")).scalar_one()
        if count == 0:
            sample = create_sample_data(120)
            for _, row in sample.iterrows():
                conn.execute(text("""INSERT INTO properties
                (city, area_sqft, bedrooms, bathrooms, house_type, furnishing_status, parking, year_built, price_lakh, status)
                VALUES (:city,:area_sqft,:bedrooms,:bathrooms,:house_type,:furnishing_status,:parking,:year_built,:price_lakh,'Available')"""),
                {k: (int(v) if k in ["area_sqft","bedrooms","bathrooms","parking","year_built"] else float(v) if k=="price_lakh" else str(v))
                 for k, v in row.to_dict().items()})

def get_training_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
        if missing:
            raise ValueError("CSV is missing required columns: " + ", ".join(missing))
        return df[FEATURES + [TARGET]].dropna(subset=[TARGET])
    return create_sample_data(1000)

def train_model(force=False):
    if os.path.exists(MODEL_PATH) and not force:
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            pass
    df = get_training_data()
    X, y = df[FEATURES], df[TARGET]
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                 ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    preprocessor = ColumnTransformer([("num", numeric_pipe, NUMERIC_FEATURES),
                                      ("cat", categorical_pipe, CATEGORICAL_FEATURES)])
    model = Pipeline([("preprocessor", preprocessor),
                      ("regressor", RandomForestRegressor(n_estimators=180, random_state=42, min_samples_leaf=2))])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {"mae": round(float(mean_absolute_error(y_test, pred)), 2),
               "rmse": round(float(np.sqrt(mean_squared_error(y_test, pred))), 2),
               "r2": round(float(r2_score(y_test, pred)), 3)}
    joblib.dump({"model": model, "metrics": metrics, "features": FEATURES}, MODEL_PATH)
    return {"model": model, "metrics": metrics, "features": FEATURES}

def query_df(sql, params=None):
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


# Chart PNGs are generated from the live database and kept in memory.
CHART_IMAGES = {}

def make_charts():
    df = query_df("""
        SELECT city, area_sqft, bedrooms, bathrooms, house_type,
               furnishing_status, year_built, price_lakh
        FROM properties
    """)
    if df.empty:
        CHART_IMAGES.clear()
        return

    bg, panel, grid, textc = "#061b43", "#082653", "#174779", "#eef6ff"
    palette = ["#08c8ff", "#a855f7", "#f43f9e", "#ffbf24", "#10e6a1", "#6477ff"]
    sns.set_theme(style="darkgrid")

    def finish(fig, name):
        fig.patch.set_facecolor(bg)
        fig.tight_layout(pad=2)
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=160, facecolor=bg,
                    bbox_inches="tight", edgecolor=bg)
        buf.seek(0)
        CHART_IMAGES[name] = buf.getvalue()
        plt.close(fig)

    def style(ax, title, xlabel=None, ylabel=None):
        ax.set_facecolor(panel)
        ax.set_title(title, color=textc, fontsize=15, fontweight="bold", pad=15)
        if xlabel: ax.set_xlabel(xlabel, color=textc, fontsize=11)
        if ylabel: ax.set_ylabel(ylabel, color=textc, fontsize=11)
        ax.tick_params(colors="#d6e8ff", labelsize=9)
        ax.grid(True, color=grid, alpha=.75, linestyle="--")
        for sp in ax.spines.values(): sp.set_color("#1685d7")

    # 1. Price distribution with neon histogram and KDE
    fig, ax = plt.subplots(figsize=(9, 5))
    counts, edges, bars = ax.hist(df["price_lakh"], bins=18, edgecolor=bg, linewidth=1.2)
    for i, bar in enumerate(bars):
        bar.set_facecolor(palette[i % len(palette)])
        bar.set_alpha(.95)
    sns.kdeplot(df["price_lakh"], ax=ax, color="#29d8ff", linewidth=2.8)
    peak = int(np.argmax(counts)) if len(counts) else 0
    if len(bars):
        bars[peak].set_edgecolor("#ffe45c")
        bars[peak].set_linewidth(2.5)
    style(ax, "PRICE DISTRIBUTION", "Price (₹ lakh)", "Number of properties")
    finish(fig, "price_distribution.png")

    # 2. Price vs area scatter with trend line
    fig, ax = plt.subplots(figsize=(9, 5))
    cities = list(df["city"].dropna().unique())
    city_colors = {c: palette[i % len(palette)] for i, c in enumerate(cities)}
    for city, group in df.groupby("city"):
        ax.scatter(group["area_sqft"], group["price_lakh"], s=42, alpha=.88,
                   color=city_colors[city], edgecolors="#e7f4ff", linewidths=.35)
    sns.regplot(data=df, x="area_sqft", y="price_lakh", scatter=False, ax=ax,
                color="#ff72c6", line_kws={"linewidth": 3})
    style(ax, "PRICE VS AREA", "Area (sq ft)", "Price (₹ lakh)")
    finish(fig, "price_vs_area.png")

    # 3. City average price horizontal bars with value labels
    by_city = df.groupby("city")["price_lakh"].mean().sort_values(ascending=False).head(6).sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(by_city.index, by_city.values, color=[palette[i % len(palette)] for i in range(len(by_city))],
                   edgecolor="#eaf6ff", linewidth=.5, height=.65)
    for bar, val in zip(bars, by_city.values):
        ax.text(val + max(by_city.values)*.012, bar.get_y()+bar.get_height()/2,
                f"₹ {val:.1f} L", va="center", color="#f4f8ff", fontsize=10, fontweight="bold")
    ax.set_xlim(0, max(by_city.values)*1.23)
    style(ax, "AVERAGE PRICE BY CITY", "Average price (₹ lakh)", "City")
    finish(fig, "city_prices.png")

    # 4. House type composition donut/pie
    type_counts = df["house_type"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    wedges, labels, pct = ax.pie(type_counts.values, labels=type_counts.index,
        autopct="%1.0f%%", startangle=90, colors=palette[:len(type_counts)],
        pctdistance=.70, labeldistance=1.12,
        wedgeprops={"edgecolor": bg, "linewidth": 3},
        textprops={"color": textc, "fontsize": 10, "weight": "bold"})
    for t in pct: t.set_color("#ffffff"); t.set_fontsize(12); t.set_weight("bold")
    ax.set_title("HOUSE TYPE DISTRIBUTION", color=textc, fontsize=15, fontweight="bold", pad=15)
    ax.set_facecolor(bg)
    finish(fig, "house_types.png")

    # 5. Average price by bedroom count, labelled bars
    by_bed = df.groupby("bedrooms")["price_lakh"].mean().sort_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(by_bed.index.astype(str), by_bed.values,
                  color=[palette[i % len(palette)] for i in range(len(by_bed))],
                  edgecolor="#eaf6ff", linewidth=.6, width=.78)
    for bar, val in zip(bars, by_bed.values):
        ax.text(bar.get_x()+bar.get_width()/2, val+max(by_bed.values)*.018,
                f"₹ {val:.1f} L", ha="center", va="bottom",
                color="#eaf6ff", fontsize=10, fontweight="bold")
    ax.set_ylim(0, max(by_bed.values)*1.22)
    style(ax, "AVERAGE PRICE BY BEDROOMS", "Bedrooms", "Average price (₹ lakh)")
    finish(fig, "bedrooms.png")


@app.route("/charts/<filename>")
def chart_image(filename):
    """Serve the original chart PNG files stored in static/charts."""
    allowed = {
        "price_distribution.png",
        "price_vs_area.png",
        "city_prices.png",
        "house_types.png",
        "bedrooms.png",
    }

    if filename not in allowed:
        return "Chart not found", 404

    image_path = os.path.join(app.static_folder, "charts", filename)
    if not os.path.isfile(image_path):
        return "Chart unavailable", 404

    return send_file(image_path, mimetype="image/png", conditional=True)

def metrics():
    data = query_df("SELECT COUNT(*) AS n, AVG(price_lakh) AS avg_price, AVG(area_sqft) AS avg_area FROM properties")
    return data.iloc[0].to_dict() if not data.empty else {"n": 0, "avg_price": 0, "avg_area": 0}

@app.route("/")
def home():
    try:
        stats = metrics()
        recent = query_df("SELECT * FROM prediction_history ORDER BY id DESC LIMIT 5")
        model_data = train_model()
        return render_template("index.html", stats=stats, recent=recent.to_dict("records"),
                               model_metrics=model_data["metrics"])
    except Exception as exc:
        return render_template("error.html", error=str(exc)), 500

@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    form = {"city":"Hyderabad", "area_sqft":"1200", "bedrooms":"3", "bathrooms":"2",
            "house_type":"Apartment", "furnishing_status":"Semi-Furnished", "parking":"1", "year_built":"2018"}
    if request.method == "POST":
        try:
            form = {k: request.form.get(k, "") for k in ["city","area_sqft","bedrooms","bathrooms","house_type","furnishing_status","parking","year_built"]}
            row = {
                "city": form["city"], "area_sqft": int(form["area_sqft"]),
                "bedrooms": int(form["bedrooms"]), "bathrooms": int(form["bathrooms"]),
                "house_type": form["house_type"], "furnishing_status": form["furnishing_status"],
                "parking": int(form["parking"]), "year_built": int(form["year_built"])
            }
            if row["area_sqft"] < 100 or row["area_sqft"] > 20000:
                raise ValueError("Area must be between 100 and 20,000 sq ft.")
            model_data = train_model()
            estimate = max(0, float(model_data["model"].predict(pd.DataFrame([row])[FEATURES])[0]))
            low, high = estimate * .88, estimate * 1.12
            with engine.begin() as conn:
                conn.execute(text("""INSERT INTO prediction_history
                (city,area_sqft,bedrooms,bathrooms,house_type,furnishing_status,parking,year_built,predicted_price)
                VALUES (:city,:area_sqft,:bedrooms,:bathrooms,:house_type,:furnishing_status,:parking,:year_built,:predicted_price)"""),
                {**row, "predicted_price": estimate})
            result = {"price": round(estimate, 2), "low": round(low, 2), "high": round(high, 2)}
        except Exception as exc:
            flash(f"Could not predict price: {exc}", "error")
    return render_template("predict.html", form=form, result=result)

@app.route("/analysis")
def analysis():
    data = query_df("SELECT city, house_type, furnishing_status, area_sqft, bedrooms, bathrooms, parking, year_built, price_lakh FROM properties")
    corr_cols = ["area_sqft","bedrooms","bathrooms","parking","year_built","price_lakh"]
    corr = data[corr_cols].corr(numeric_only=True).round(2).to_dict() if not data.empty else {}
    return render_template("analysis.html", corr=corr, columns=corr_cols)

@app.route("/houses")
def houses():
    q = request.args.get("q", "").strip()
    if q:
        sql = """SELECT * FROM properties WHERE LOWER(city) LIKE LOWER(:q) OR LOWER(house_type) LIKE LOWER(:q) ORDER BY id DESC"""
        data = query_df(sql, {"q": f"%{q}%"})
    else:
        data = query_df("SELECT * FROM properties ORDER BY id DESC")
    return render_template("houses.html", houses=data.to_dict("records"), q=q)

@app.route("/houses/add", methods=["POST"])
def add_house():
    try:
        row = read_house_form(request.form)
        with engine.begin() as conn:
            conn.execute(text("""INSERT INTO properties
            (city,area_sqft,bedrooms,bathrooms,house_type,furnishing_status,parking,year_built,price_lakh,status)
            VALUES (:city,:area_sqft,:bedrooms,:bathrooms,:house_type,:furnishing_status,:parking,:year_built,:price_lakh,:status)"""), row)
        flash("Property added successfully.", "success")
    except Exception as exc:
        flash(f"Could not add property: {exc}", "error")
    return redirect(url_for("houses"))

def read_house_form(form):
    row = {"city": form["city"].strip(), "area_sqft": int(form["area_sqft"]),
           "bedrooms": int(form["bedrooms"]), "bathrooms": int(form["bathrooms"]),
           "house_type": form["house_type"], "furnishing_status": form["furnishing_status"],
           "parking": int(form["parking"]), "year_built": int(form["year_built"]),
           "price_lakh": float(form["price_lakh"]), "status": form.get("status", "Available")}
    if not row["city"] or row["area_sqft"] < 100 or row["price_lakh"] <= 0:
        raise ValueError("Enter a city, valid area, and positive price.")
    return row

@app.route("/houses/delete/<int:house_id>", methods=["POST"])
def delete_house(house_id):
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM properties WHERE id=:id"), {"id": house_id})
    flash("Property deleted.", "success")
    return redirect(url_for("houses"))

@app.route("/history")
def history():
    data = query_df("SELECT * FROM prediction_history ORDER BY id DESC LIMIT 200")
    return render_template("history.html", history=data.to_dict("records"))

@app.route("/history/delete/<int:prediction_id>", methods=["POST"])
def delete_prediction(prediction_id):
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM prediction_history WHERE id=:id"), {"id": prediction_id})
    flash("Prediction record deleted.", "success")
    return redirect(url_for("history"))

@app.route("/dataset")
def dataset():
    data = query_df("SELECT * FROM properties ORDER BY id DESC LIMIT 250")
    return render_template("dataset.html", rows=data.to_dict("records"))

@app.route("/dataset/upload", methods=["POST"])
def upload_dataset():
    file = request.files.get("csv_file")
    if not file or not file.filename.lower().endswith(".csv"):
        flash("Please choose a CSV file.", "error")
        return redirect(url_for("dataset"))
    try:
        df = pd.read_csv(file)
        required = FEATURES + [TARGET]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError("CSV columns missing: " + ", ".join(missing))
        df = df[required].dropna(subset=[TARGET])
        with engine.begin() as conn:
            for _, r in df.iterrows():
                conn.execute(text("""INSERT INTO properties
                (city,area_sqft,bedrooms,bathrooms,house_type,furnishing_status,parking,year_built,price_lakh,status)
                VALUES (:city,:area_sqft,:bedrooms,:bathrooms,:house_type,:furnishing_status,:parking,:year_built,:price_lakh,'Available')"""),
                {k: (int(r[k]) if k in ["area_sqft","bedrooms","bathrooms","parking","year_built"] else float(r[k]) if k==TARGET else str(r[k]))
                 for k in required})
        train_model(force=True)
        flash(f"Imported {len(df)} rows and retrained the model.", "success")
    except Exception as exc:
        flash(f"CSV import failed: {exc}", "error")
    return redirect(url_for("dataset"))

@app.route("/model", methods=["GET", "POST"])
def model_page():
    if request.method == "POST":
        try:
            data = train_model(force=True)
            flash("Model retrained successfully.", "success")
        except Exception as exc:
            flash(f"Model training failed: {exc}", "error")
    data = train_model()
    return render_template("model.html", metrics=data["metrics"], features=FEATURES)

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/health")
def health():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "ok", "database": engine.dialect.name}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}, 500

with app.app_context():
    try:
        initialize_database()
        train_model()
    except Exception as startup_error:
        print("Startup setup warning:", startup_error)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=os.getenv("FLASK_DEBUG", "0") == "1")
