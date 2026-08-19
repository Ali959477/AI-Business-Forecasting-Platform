import requests,pandas as pd,streamlit as st,plotly.express as px
API='http://127.0.0.1:8000'; st.set_page_config(page_title='AI Business Forecasting',page_icon='📊',layout='wide')
st.title('🚀 AI-Powered Business Forecasting Platform'); st.caption('Retail analytics • Forecasting • Anomaly detection • Recommendations')
periods=st.sidebar.slider('Forecast months',1,24,6); up=st.sidebar.file_uploader('Upload another CSV',type='csv')
if up:
    r=requests.post(API+'/upload',files={'file':(up.name,up.getvalue(),'text/csv')}); st.sidebar.success('Dataset uploaded.') if r.ok else st.sidebar.error(r.text)
def get(e,p=None):
    try: r=requests.get(API+e,params=p,timeout=60); r.raise_for_status(); return r.json()
    except Exception as e: st.error(f'Backend error: {e}'); st.stop()
if not get('/health')['dataset_loaded']: st.error('Dataset not found. Put retail_sales.csv in the data folder.'); st.stop()
s=get('/summary'); a,b,c,d=st.columns(4); a.metric('💰 Total Sales',f"{s['total_sales']:,.2f}"); b.metric('📈 Profit','N/A'); c.metric('📦 Quantity',f"{s['total_quantity']:,.0f}"); d.metric('🧾 Orders',f"{s['total_orders']:,}")
m=pd.DataFrame(get('/monthly-sales')); m.Date=pd.to_datetime(m.Date); st.subheader('📈 Monthly Sales Trend'); st.plotly_chart(px.line(m,x='Date',y='Sales',markers=True),use_container_width=True)
x,y=st.columns(2)
with x: st.subheader('🏷️ Category Sales'); st.plotly_chart(px.bar(pd.DataFrame(get('/category-sales')),x='Category',y='Sales'),use_container_width=True)
with y: st.subheader('🌍 Regional Sales'); st.plotly_chart(px.bar(pd.DataFrame(get('/region-sales')),x='Region',y='Sales'),use_container_width=True)
st.subheader('🔮 Future Sales Forecast'); f=pd.DataFrame(get('/forecast',{'periods':periods})); f.Date=pd.to_datetime(f.Date); st.plotly_chart(px.line(f,x='Date',y='Forecast',markers=True),use_container_width=True); st.dataframe(f,use_container_width=True)
st.subheader('🚨 Anomalies'); an=pd.DataFrame(get('/anomalies')); st.dataframe(an,use_container_width=True) if not an.empty else st.success('No major anomalies detected.')
st.subheader('🤖 Business Recommendations'); [st.info(f'{i}. {r}') for i,r in enumerate(get('/recommendations')['recommendations'],1)]
