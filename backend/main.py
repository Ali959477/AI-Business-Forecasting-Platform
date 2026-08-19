from pathlib import Path
import joblib, numpy as np, pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

ROOT=Path(__file__).resolve().parent.parent
DATA_PATH=ROOT/'data'/'retail_sales.csv'
MODEL_PATH=ROOT/'models'/'sales_forecasting_model.pkl'
app=FastAPI(title='AI Business Forecasting API',version='1.0')
app.add_middleware(CORSMiddleware,allow_origins=['*'],allow_credentials=True,allow_methods=['*'],allow_headers=['*'])
DATA=None; MODEL=None
FEATURES=['Year','Month','Quarter','Time_Index','Lag_1','Lag_2','Lag_3','Rolling_Mean_3']

def load_data():
    global DATA
    if not DATA_PATH.exists(): return None
    d=pd.read_csv(DATA_PATH)
    d=d.rename(columns={'OrderDate':'Order Date','SalesAmount':'Sales','ProductCategory':'Category','OrderID':'Order ID','DiscountPct':'Discount'})
    d['Order Date']=pd.to_datetime(d['Order Date'],errors='coerce')
    DATA=d
    return d

def get_data():
    global DATA
    if DATA is None: load_data()
    if DATA is None: raise HTTPException(404,'Dataset not found.')
    return DATA

def monthly(d):
    x=d.dropna(subset=['Order Date','Sales']).copy()
    m=x.groupby(pd.Grouper(key='Order Date',freq='MS'))['Sales'].sum().reset_index()
    m.columns=['Date','Sales']; m['Year']=m.Date.dt.year; m['Month']=m.Date.dt.month; m['Quarter']=m.Date.dt.quarter; m['Time_Index']=np.arange(len(m))
    m['Lag_1']=m.Sales.shift(1); m['Lag_2']=m.Sales.shift(2); m['Lag_3']=m.Sales.shift(3); m['Rolling_Mean_3']=m.Sales.rolling(3).mean()
    return m.dropna().reset_index(drop=True)

def get_model():
    global MODEL
    if MODEL is not None: return MODEL
    if MODEL_PATH.exists(): MODEL=joblib.load(MODEL_PATH); return MODEL
    from sklearn.ensemble import RandomForestRegressor
    m=monthly(get_data()); MODEL=RandomForestRegressor(n_estimators=300,random_state=42).fit(m[FEATURES],m.Sales); joblib.dump(MODEL,MODEL_PATH); return MODEL

@app.get('/')
def root(): return {'message':'AI Business Forecasting API is running'}
@app.get('/health')
def health(): return {'status':'healthy','dataset_loaded':DATA is not None or DATA_PATH.exists(),'model_loaded':MODEL is not None or MODEL_PATH.exists()}
@app.post('/upload')
async def upload(file:UploadFile=File(...)):
    global DATA,MODEL
    DATA_PATH.write_bytes(await file.read()); DATA=MODEL=None; load_data(); return {'message':'Dataset uploaded','rows':len(DATA),'columns':DATA.columns.tolist()}
@app.get('/summary')
def summary():
    d=get_data(); return {'rows':len(d),'columns':len(d.columns),'total_sales':float(d.Sales.sum()),'total_profit':0.0,'total_quantity':float(d.Quantity.sum()) if 'Quantity' in d else 0,'total_orders':int(d['Order ID'].nunique()) if 'Order ID' in d else len(d)}
@app.get('/monthly-sales')
def monthly_sales():
    d=get_data(); r=d.groupby(pd.Grouper(key='Order Date',freq='MS')).Sales.sum().reset_index(); r['Order Date']=r['Order Date'].dt.strftime('%Y-%m-%d'); return r.rename(columns={'Order Date':'Date'}).to_dict('records')
@app.get('/category-sales')
def category_sales(): return get_data().groupby('Category').Sales.sum().sort_values(ascending=False).reset_index().to_dict('records')
@app.get('/region-sales')
def region_sales(): return get_data().groupby('Region').Sales.sum().sort_values(ascending=False).reset_index().to_dict('records')
@app.get('/forecast')
def forecast(periods:int=6):
    if not 1<=periods<=24: raise HTTPException(400,'Periods must be 1-24')
    model=get_model(); h=monthly(get_data()); out=[]
    for _ in range(periods):
        date=h.Date.iloc[-1]+pd.offsets.MonthBegin(1); row={'Date':date,'Year':date.year,'Month':date.month,'Quarter':date.quarter,'Time_Index':len(h),'Lag_1':h.Sales.iloc[-1],'Lag_2':h.Sales.iloc[-2],'Lag_3':h.Sales.iloc[-3],'Rolling_Mean_3':h.Sales.tail(3).mean()}; pred=float(model.predict(pd.DataFrame([row])[FEATURES])[0]); row['Sales']=pred; h=pd.concat([h,pd.DataFrame([row])],ignore_index=True); out.append({'Date':date.strftime('%Y-%m-%d'),'Forecast':pred})
    return out
@app.get('/anomalies')
def anomalies():
    m=monthly(get_data()); m['Mean']=m.Sales.rolling(6).mean(); m['Std']=m.Sales.rolling(6).std(); m['Upper']=m.Mean+2*m.Std; m['Lower']=m.Mean-2*m.Std; m['Anomaly']=(m.Sales>m.Upper)|(m.Sales<m.Lower); r=m[m.Anomaly].copy(); r['Date']=r.Date.dt.strftime('%Y-%m-%d'); return r[['Date','Sales','Upper','Lower','Anomaly']].replace({np.nan:None}).to_dict('records')
@app.get('/recommendations')
def recommendations():
    d=get_data(); rec=[]
    rec.append(f"Prioritize inventory for {d.groupby('Category').Sales.sum().idxmax()}, the top sales category.")
    rec.append(f"Review {d.groupby('Region').Sales.sum().idxmin()}, the lowest-sales region.")
    if 'Discount' in d and len(d): rec.append('Monitor discount levels and their effect on sales performance.')
    n=len(anomalies()); rec.append(f'Investigate {n} detected monthly sales anomalies.' if n else 'No major monthly sales anomalies were detected.')
    rec.append('Use forecasts to improve inventory and purchasing decisions.')
    return {'recommendations':rec}
