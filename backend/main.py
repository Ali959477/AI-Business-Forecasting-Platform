from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware


# =========================================================
# PATHS
# =========================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "retail_sales.csv"
MODEL_PATH = ROOT / "models" / "sales_forecasting_model.pkl"


# =========================================================
# APP
# =========================================================

app = FastAPI(
    title="AI Business Forecasting API",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# =========================================================
# GLOBAL VARIABLES
# =========================================================

DATA = None
MODEL = None

FEATURES = [
    "Year",
    "Month",
    "Quarter",
    "Time_Index",
    "Lag_1",
    "Lag_2",
    "Lag_3",
    "Rolling_Mean_3"
]


# =========================================================
# LOAD DATA
# =========================================================

def load_data():
    global DATA

    if not DATA_PATH.exists():
        return None

    # Fix CSV encoding problem
    try:
        d = pd.read_csv(DATA_PATH, encoding="utf-8")
    except UnicodeDecodeError:
        d = pd.read_csv(DATA_PATH, encoding="latin1")

    # Rename columns
    d = d.rename(columns={
        "OrderDate": "Order Date",
        "SalesAmount": "Sales",
        "ProductCategory": "Category",
        "OrderID": "Order ID",
        "DiscountPct": "Discount"
    })

    # Convert date
    if "Order Date" in d.columns:
        d["Order Date"] = pd.to_datetime(
            d["Order Date"],
            errors="coerce"
        )

    DATA = d

    return d


# =========================================================
# GET DATA
# =========================================================

def get_data():
    global DATA

    if DATA is None:
        load_data()

    if DATA is None:
        raise HTTPException(
            status_code=404,
            detail="Dataset not found."
        )

    return DATA


# =========================================================
# MONTHLY DATA
# =========================================================

def monthly(d):

    if "Order Date" not in d.columns:
        raise HTTPException(
            status_code=500,
            detail="Order Date column not found in dataset."
        )

    if "Sales" not in d.columns:
        raise HTTPException(
            status_code=500,
            detail="Sales column not found in dataset."
        )

    x = d.dropna(
        subset=["Order Date", "Sales"]
    ).copy()

    m = (
        x.groupby(
            pd.Grouper(
                key="Order Date",
                freq="MS"
            )
        )["Sales"]
        .sum()
        .reset_index()
    )

    m.columns = ["Date", "Sales"]

    m["Year"] = m["Date"].dt.year
    m["Month"] = m["Date"].dt.month
    m["Quarter"] = m["Date"].dt.quarter

    m["Time_Index"] = np.arange(len(m))

    m["Lag_1"] = m["Sales"].shift(1)
    m["Lag_2"] = m["Sales"].shift(2)
    m["Lag_3"] = m["Sales"].shift(3)

    m["Rolling_Mean_3"] = (
        m["Sales"]
        .rolling(3)
        .mean()
    )

    return m.dropna().reset_index(drop=True)


# =========================================================
# GET MODEL
# =========================================================

def get_model():

    global MODEL

    if MODEL is not None:
        return MODEL

    # Load existing model
    if MODEL_PATH.exists():

        try:
            MODEL = joblib.load(MODEL_PATH)
            return MODEL

        except Exception:
            MODEL = None

    # Train new model if model doesn't exist
    from sklearn.ensemble import RandomForestRegressor

    m = monthly(get_data())

    if len(m) < 5:
        raise HTTPException(
            status_code=500,
            detail="Not enough data to train forecasting model."
        )

    MODEL = RandomForestRegressor(
        n_estimators=300,
        random_state=42
    )

    MODEL.fit(
        m[FEATURES],
        m["Sales"]
    )

    # Make sure models folder exists
    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        MODEL,
        MODEL_PATH
    )

    return MODEL


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():

    return {
        "message": "AI Business Forecasting API is running"
    }


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "dataset_loaded": (
            DATA is not None
            or DATA_PATH.exists()
        ),
        "model_loaded": (
            MODEL is not None
            or MODEL_PATH.exists()
        )
    }


# =========================================================
# UPLOAD DATASET
# =========================================================

@app.post("/upload")
async def upload(
    file: UploadFile = File(...)
):

    global DATA
    global MODEL

    try:

        content = await file.read()

        DATA_PATH.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        DATA_PATH.write_bytes(content)

        DATA = None
        MODEL = None

        loaded_data = load_data()

        if loaded_data is None:
            raise HTTPException(
                status_code=500,
                detail="Uploaded dataset could not be loaded."
            )

        return {
            "message": "Dataset uploaded successfully",
            "rows": len(loaded_data),
            "columns": loaded_data.columns.tolist()
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Upload error: {str(e)}"
        )


# =========================================================
# SUMMARY
# =========================================================

@app.get("/summary")
def summary():

    d = get_data()

    return {
        "rows": int(len(d)),
        "columns": int(len(d.columns)),

        "total_sales": (
            float(d["Sales"].sum())
            if "Sales" in d.columns
            else 0.0
        ),

        "total_profit": (
            float(d["Profit"].sum())
            if "Profit" in d.columns
            else 0.0
        ),

        "total_quantity": (
            float(d["Quantity"].sum())
            if "Quantity" in d.columns
            else 0.0
        ),

        "total_orders": (
            int(d["Order ID"].nunique())
            if "Order ID" in d.columns
            else int(len(d))
        )
    }


# =========================================================
# MONTHLY SALES
# =========================================================

@app.get("/monthly-sales")
def monthly_sales():

    d = get_data()

    if "Order Date" not in d.columns:
        raise HTTPException(
            status_code=500,
            detail="Order Date column not found."
        )

    if "Sales" not in d.columns:
        raise HTTPException(
            status_code=500,
            detail="Sales column not found."
        )

    r = (
        d.groupby(
            pd.Grouper(
                key="Order Date",
                freq="MS"
            )
        )["Sales"]
        .sum()
        .reset_index()
    )

    r["Order Date"] = (
        r["Order Date"]
        .dt.strftime("%Y-%m-%d")
    )

    r = r.rename(
        columns={
            "Order Date": "Date"
        }
    )

    return r.to_dict("records")


# =========================================================
# CATEGORY SALES
# =========================================================

@app.get("/category-sales")
def category_sales():

    d = get_data()

    if "Category" not in d.columns:
        return []

    return (
        d.groupby("Category")["Sales"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
        .to_dict("records")
    )


# =========================================================
# REGION SALES
# =========================================================

@app.get("/region-sales")
def region_sales():

    d = get_data()

    if "Region" not in d.columns:
        return []

    return (
        d.groupby("Region")["Sales"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
        .to_dict("records")
    )


# =========================================================
# FORECAST
# =========================================================

@app.get("/forecast")
def forecast(periods: int = 6):

    if not 1 <= periods <= 24:

        raise HTTPException(
            status_code=400,
            detail="Periods must be between 1 and 24."
        )

    model = get_model()

    h = monthly(get_data())

    if len(h) < 3:

        raise HTTPException(
            status_code=500,
            detail="Not enough monthly data for forecasting."
        )

    output = []

    for _ in range(periods):

        date = (
            h["Date"].iloc[-1]
            + pd.offsets.MonthBegin(1)
        )

        row = {

            "Date": date,

            "Year": date.year,

            "Month": date.month,

            "Quarter": date.quarter,

            "Time_Index": len(h),

            "Lag_1": h["Sales"].iloc[-1],

            "Lag_2": h["Sales"].iloc[-2],

            "Lag_3": h["Sales"].iloc[-3],

            "Rolling_Mean_3": (
                h["Sales"]
                .tail(3)
                .mean()
            )
        }

        prediction = float(
            model.predict(
                pd.DataFrame([row])[FEATURES]
            )[0]
        )

        row["Sales"] = prediction

        h = pd.concat(
            [
                h,
                pd.DataFrame([row])
            ],
            ignore_index=True
        )

        output.append(
            {
                "Date": date.strftime("%Y-%m-%d"),
                "Forecast": prediction
            }
        )

    return output


# =========================================================
# ANOMALIES
# =========================================================

@app.get("/anomalies")
def anomalies():

    m = monthly(get_data())

    m["Mean"] = (
        m["Sales"]
        .rolling(6)
        .mean()
    )

    m["Std"] = (
        m["Sales"]
        .rolling(6)
        .std()
    )

    m["Upper"] = (
        m["Mean"]
        + 2 * m["Std"]
    )

    m["Lower"] = (
        m["Mean"]
        - 2 * m["Std"]
    )

    m["Anomaly"] = (
        (m["Sales"] > m["Upper"])
        |
        (m["Sales"] < m["Lower"])
    )

    r = m[m["Anomaly"]].copy()

    r["Date"] = (
        r["Date"]
        .dt.strftime("%Y-%m-%d")
    )

    return (
        r[
            [
                "Date",
                "Sales",
                "Upper",
                "Lower",
                "Anomaly"
            ]
        ]
        .replace({np.nan: None})
        .to_dict("records")
    )


# =========================================================
# RECOMMENDATIONS
# =========================================================

@app.get("/recommendations")
def recommendations():

    d = get_data()

    rec = []

    # Top category
    if "Category" in d.columns:

        top_category = (
            d.groupby("Category")["Sales"]
            .sum()
            .idxmax()
        )

        rec.append(
            f"Prioritize inventory for {top_category}, "
            "the top sales category."
        )

    # Lowest region
    if "Region" in d.columns:

        lowest_region = (
            d.groupby("Region")["Sales"]
            .sum()
            .idxmin()
        )

        rec.append(
            f"Review {lowest_region}, "
            "the lowest-sales region."
        )

    # Discount
    if "Discount" in d.columns and len(d):

        rec.append(
            "Monitor discount levels and their "
            "effect on sales performance."
        )

    # Anomalies
    try:

        anomaly_data = anomalies()

        n = len(anomaly_data)

        if n:
            rec.append(
                f"Investigate {n} detected "
                "monthly sales anomalies."
            )
        else:
            rec.append(
                "No major monthly sales anomalies "
                "were detected."
            )

    except Exception:

        rec.append(
            "Review monthly sales for unusual patterns."
        )

    # General recommendation
    rec.append(
        "Use forecasts to improve inventory "
        "and purchasing decisions."
    )

    return {
        "recommendations": rec
    }
