# AI Business Forecasting Platform - Fixed for Uploaded Dataset

Dataset is already included in `data/retail_sales.csv`.

Run from the project root:

```bash
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

Open a second terminal:

```bash
streamlit run frontend/app.py
```

Dashboard: http://localhost:8501
API docs: http://127.0.0.1:8000/docs

The uploaded dataset contains OrderDate, ProductCategory, Region, Quantity, SalesAmount, DiscountPct, etc. The backend automatically maps these to the dashboard's required fields.
