# our streamlit app to serve as the frontend for everything

import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import numpy as np
from prophet.serialize import model_to_json, model_from_json
from prophet.plot import plot_plotly, plot_components_plotly

st.set_page_config(layout="wide", page_title="Narragansett Bay Time Series Data Visualization and Modeling", page_icon="🌊")

# Anchored to this file rather than the working directory. Streamlit Community
# Cloud runs every app from the repository root even when the entrypoint lives
# in a subdirectory, so '../data/...' would resolve outside the repo there.
# sample_data/ is the committed synthetic set the deployment reads; data/ is
# gitignored and only exists on machines that ran make_sample_data.py.
DATA_DIR = Path(__file__).resolve().parent.parent / "sample_data"

forecast_csv = pd.read_csv(DATA_DIR / "GD_forecast.csv")
with open(DATA_DIR / "serialized_model.json", "r") as fin:
    m = model_from_json(fin.read())
    

#style = "<style>h2 {text-align: center;} h1 {text-align: center;} body {text-align: center;}</style>"
#st.markdown(style, unsafe_allow_html=True)

df = pd.read_csv(DATA_DIR / "Daily_Means_Through_2022.csv")
# date to datetime (though we change it again after so idk if this is even smart)
df['date'] = pd.to_datetime(df['date'])
df_pivot = df.pivot_table(index=['date', 'site', 'depth'], columns='parameter', values='measure', aggfunc='mean')
df_pivot.reset_index(inplace=True)
#df_pivot.dropna(inplace=True)

# make a title

# Optional access gate. Set APP_PASSWORD in the environment or in
# .streamlit/secrets.toml to require one. Left unset the app is open, which is
# what you want when it is serving the synthetic sample data in this repo.
expected_password = os.environ.get("APP_PASSWORD")
try:
    expected_password = st.secrets.get("APP_PASSWORD", expected_password)
except Exception:
    pass  # no secrets.toml present

if expected_password:
    pwd = st.text_input("Enter the password:", type="password")
    if pwd != expected_password:
        st.stop()
    st.write("Password accepted")


st.title("Narragansett Bay Time Series Data Visualization and Modeling 🌊")
st.write("_By Siddharth Gupta_")

st.warning('''**This deployment runs on synthetic sample data.** The real Fixed-Site Monitoring
           Network measurements belong to RI DEM, the Narragansett Bay Commission and URI GSO,
           and are not redistributed in this repository. Every number below was fabricated by
           `make_sample_data.py` and says nothing about the actual state of the bay.''')


st.header("Introduction")
st.write('''This is a Streamlit app that allows you to explore the Narragansett Bay Fixed-Site Monitoring Stations' Data. 
         The data is collected by the Rhode Island Department of Environmental Management in association with other agencies such as 
         the Narragansett Bay Commission and URI Graduate School of Oceanography.''')
st.subheader("Why is this data important?")
st.write('''The data collected by these sites is important in monitoring the health of the bay. **Dissolved oxygen**, 
         one of the most important indicators, is essential to support life in the water. 
         Low dissolved oxygen concentration can lead to issues like fish kills. By measuring it
         and other indicators (water temperature, salinity, etc...), the DEM and its partners are able
         to identify if the bay is hypoxic (particularly low in dissolved oxygen).''')
st.header("About the Data")
st.write('''Here's what the data looks like in an Excel-style format. Note that we are working with daily averages of the data,
         and measurements from the buoys are actually taken every 15 minutes.''')
st.dataframe(df_pivot)
st.subheader("Station code/abbreviation key")
st.markdown('''| Station Code | Location Description | Abbreviation |
|-------------|----------------------|-------------|
| B2          | North of Prudence Island (representative of Upper Bay) | NP |
| B3          | South of Conimicut Point (station just south of lighthouse) | CP |
| B3W         | Upper Bay Winter Station (station on channel Marker 13 just south of Conimicut Pt) | UB |
| B4          | Bullock's Neck or Reach/Lower Providence River (downstream of Fields Point Wastewater Treatment Facility) | BR |
| B6          | Mount View (mouth of Greenwich Bay) | MV |
| B7          | Quonset Point | QP |
| B10         | Cole River (mouth of Cole River in Mt Hope Bay) | CR |
| B11         | Taunton River (Mouth of Taunton River in Mt Hope Bay) | TR |
| B12         | Mount Hope Bay | MH |
| B13         | Poppasquash Point (Upper East Passage) | PP |
| B14         | Sally Rock (Mid-Greenwich Bay) | SR |
| F3          | T-Wharf (South of Prudence Island on East Passage) | TW |
| F4          | Phillipsdale (in Seekonk River-downstream of Bucklin WWTF) | PD |
| F5          | Greenwich Bay (western edge in a marina near the mouth of Appanoug Cove) | GB |
| F7          | URI GSO Dock | GD |
''')
with st.expander("Description of data (spread, mins, maxes)"):
    st.write(df_pivot.describe())
with st.expander("Random sample of data"):
    st.write("Here's a random sample of the data as-is:")
    st.write(df_pivot.sample(20))

with st.expander("View map of collection sites"):
    st.image("https://dem.ri.gov/sites/g/files/xkgbur861/files/styles/max_1300x1300/public/2022-06/locatmap.jpg?itok=RFpJDqJx", width=400)
#with st.expander("View station code/abbreviation key"):


st.write("As you can see, the data is sometimes missing, as it's collected seasonally (but even within seasons, data is often missing). Data is presented as-is and processed as-is unless mentioned otherwise.")

st.header("Data Visualization")

# st.write("Here's some of the data, with averages between each site and each probe type every day:")
# df_pivot['site'] = 'All Sites'
# df_pivot['depth'] = 'All Depths'
# df_pivot = df_pivot.groupby(['date', 'site', 'depth']).mean().reset_index()
# df_pivot.drop(columns=['site', 'depth', "DO_pct", "Depth_m"], inplace=True)
# df_pivot['date'] = df_pivot['date'].dt.to_period('W').dt.start_time
# df_pivot = df_pivot.groupby(['date']).mean().reset_index()
# df_pivot['date'] = pd.to_datetime(df_pivot['date'])

#st.line_chart(x="date", y="Temp_C", data=df_pivot)

st.subheader("Visualizing indicators over time")

#parameter = st.selectbox('Select a parameter:', df["parameter"].unique())
parameters = st.multiselect('Select parameters:', df["parameter"].unique(), default=["DO_mg/L"])

site = st.selectbox('Select a site:', df_pivot['site'].unique())
#sites = st.multiselect('Select sites:', df_pivot['site'].unique())

depth = st.selectbox('Select a depth:', df_pivot['depth'].unique())
st.write("note that BR is the _only_ site with a mid probe")

#filtered_data = df_pivot[(df_pivot['site'] == site) & (df_pivot['depth'] == depth)]
filtered_data = df_pivot[(df_pivot['depth'] == depth) & (df_pivot['site'] == site)]
if filtered_data.empty:
    st.write("No data available for the selected parameter, site, and depth combination. Try something else!")
else:
    # insert a check for missing days
    filtered_data['date'] = pd.to_datetime(filtered_data['date'])
    full_date_range = pd.date_range(start=filtered_data['date'].min(), end=filtered_data['date'].max(), freq='D')
    filtered_data = filtered_data.set_index('date').reindex(full_date_range).reset_index()
    filtered_data.rename(columns={'index': 'date'}, inplace=True)
    
    param_list = []
    for param in parameters:
        if param not in filtered_data.columns:
            st.write(f"No data available for {param} at {site} ({depth}).")
            continue
        #st.write(f"Visualizing {param} at {site} ({depth})")
        param_list.append(param)
        
    fig = px.line(filtered_data, x='date', y=param_list, title=f'{param_list} at {site} ({depth})')
    #fig.update_traces(line=dict(color='#00A3E0'))
    #fig.update_traces(connectgaps=True)

    # begin at 2001
    fig.update_xaxes(
        dtick="M12",
        tickformat="%Y",
        ticklabelmode="period",
        range=["2000-01-01", "2026-01-01"]
    )
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Value",
        legend_title="Parameter",
        title_font_size=24,
        height=800,
        #width = 1500,
    )

    st.plotly_chart(fig)

with st.expander("Visualize a second site/parameter/depth combination", expanded=False):
    parameters_2 = st.multiselect('Select parameters:', df["parameter"].unique(), key="parameters_2")
    site_2 = st.selectbox('Select a site:', df_pivot['site'].unique(), key="site_2")
    depth_2 = st.selectbox('Select a depth:', df_pivot['depth'].unique(), key="depth_2")
    st.write("note that BR is the _only_ site with a mid probe")
    filtered_data = df_pivot[(df_pivot['depth'] == depth_2) & (df_pivot['site'] == site_2)]
    if filtered_data.empty:
        st.write("No data available for the selected parameter, site, and depth combination. Try something else!")
    else:
        filtered_data['date'] = pd.to_datetime(filtered_data['date'])
        full_date_range = pd.date_range(start=filtered_data['date'].min(), end=filtered_data['date'].max(), freq='D')
        filtered_data = filtered_data.set_index('date').reindex(full_date_range).reset_index()
        filtered_data.rename(columns={'index': 'date'}, inplace=True)
        
        param_list = []
        for param in parameters_2:
            if param not in filtered_data.columns:
                st.write(f"No data available for {param} at {site_2} ({depth_2}).")
                continue
            param_list.append(param)
            
        fig = px.line(filtered_data, x='date', y=param_list, title=f'{param_list} at {site_2} ({depth_2})')
        fig.update_xaxes(
            dtick="M12",
            tickformat="%Y",
            ticklabelmode="period",
            range=["2000-01-01", "2026-01-01"]
        )
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Value",
            legend_title="Parameter",
            title_font_size=24,
            height=800,
        )
        st.plotly_chart(fig, key="fig2")

st.subheader("How often does the bay have low oxygen levels")
st.write("Visualize the percentage of days when dissolved oxygen levels are below 5 mg/L at each site at a certain depth and year. DO levels below 5 mg/L can begin to adversely afferct organisms in the water.")
# make pie chart of days hypoxic vs not hypoxic
#site1 = st.selectbox('Select a site:', df_pivot['site'].unique(), key="site1")
depth1 = st.selectbox('Select a depth:', df_pivot['depth'].unique(), key="depth1")
year = st.selectbox('Select a year:', df_pivot['date'].dt.year.unique(), key="year")

filtered_data = df_pivot[(df_pivot['depth'] == depth1) & (df_pivot['date'].dt.year == year)]
if filtered_data.empty:
    st.write("No data available for the selected site and depth combination. Try something else!")
else:
    filtered_data['hypoxic'] = filtered_data.apply(lambda row: row['DO_mg/L'] < 5 if not pd.isnull(row["DO_mg/L"]) else "Not Collected", axis=1)
    filtered_data['hypoxic'] = filtered_data['hypoxic'].replace({True: 'Hypoxic', False: 'Not Hypoxic'})
    
    fig = px.pie(filtered_data, names='hypoxic', title=f'Hypoxia at ({depth1})', hole=0.3, facet_col='site', facet_col_wrap=4) 
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1], font_size=20))
    fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Value",
            legend_title="Parameter",
            title_font_size=24,
            height=800,
        )
    #colors = ['gold', 'mediumturquoise', 'darkorange', 'lightgreen']
    fig.update_traces(hoverinfo='label+percent', textinfo = "none" , textfont_size=15,
                  marker=dict(line=dict(color='#000000', width=2)), hovertemplate='%{label}: %{percent}', hoverlabel=dict(
    
        font_size=20,
        font_family="monospace"
    ))
    
    st.plotly_chart(fig)

st.header("Forecasting")
st.write('''After testing a series of models, we find that Facebook's Prophet model is most effective in modeling data for our continuous sites (seasonal sites were not
         found to be easily model-able). Achieving an average 5.02 percent error across years in two-week DO forecasts, our model uses past DO values along with past
         values of other indicators in its forecasting.''')
st.plotly_chart(plot_plotly(m, forecast_csv))

st.header("Final Disclaimer")
st.write('''The figures above are computed from synthetic sample data, not from observations —
         the 5.02 percent error quoted for the Prophet model comes from the original study on the
         real record, not from anything shown here. The real data is owned by RI DEM, the
         Narragansett Bay Commission and URI GSO; it is presented as-is in the private version of
         this project and should not be taken or used without express permission of its owners.
         Please contact Siddharth Gupta (emailsiddha@gmail.com) with further questions!''')