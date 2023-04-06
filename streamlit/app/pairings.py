
import sys
import re
from functools import lru_cache
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Functions
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    with open(path) as tf:
        lines = tf.readlines()
    pairs_list = []
    for l in lines:
        if 'Number of pairings' in l:
            pair = l.split('Number of pairings')[-1].strip('\n').split('is')
            pairs_list.append([p.strip(' ') for p in pair])
    df = pd.DataFrame(pairs_list, columns=['pairing', 'number'])
    return df

@st.cache_data
def read_data(txt_file: str) -> pd.DataFrame:
    pairs_list = []
    for l in txt_file:
        l = l.decode("utf-8")
        if 'Number of pairings' in l:
            pair = l.split('Number of pairings')[-1].strip('\n').split('is')
            pairs_list.append([p.strip(' ') for p in pair])
    df = pd.DataFrame(pairs_list, columns=['pairing', 'number'])
    return df


def clean_shifts(pairing:str):
    shifts = pairing.strip('[]').split('>')
    return shifts

def get_lags(list_of_shifts):
    lags = [l.split(':') for l in list_of_shifts]
    lags = [k for m in lags for k in m]
    return lags

def extract_shift_and_lag(df) -> pd.DataFrame:
    df['shift'] = df['pairing'].apply(clean_shifts)
    df['lags_list'] = df['shift'].apply(lambda x: [l.split(':') for l in x])
    return df

def find_shift_n(shifts_str, n):
    try: 
        return shifts_str[n]
    except IndexError:
        return ''


@lru_cache(maxsize=None)
def list_airports(pairing):
    return  re.findall(pattern = r'[A-Z]{3}', string = pairing)

def plot_pairings(df):
    ann_font_size = 12
    space = 40
    annotations = []

    fig = go.Figure()

    custom_ticktext = [str(i) for i in df.number.values]
    colors = ["khaki", "darkolivegreen"]
    # print(df['shift'])
    max_shifts = df['shift'].apply(lambda x: len(x)).max()
    # print('max_shifts', max_shifts)
    pos_series = pd.Series([20]*len(df), index = df.index)
    for i in range(max_shifts):
        color_id = 1
        if i%2 == 0:
            color_id = 1
        shift_ser = df['shift'].apply(lambda x: find_shift_n(x, n=i))
        len_ser = shift_ser.apply(lambda x: len(x.split(':')) if x!= '' else 0)

        fig.add_trace(go.Bar(
            y=[i for i in range(len(df))],
            x=(len_ser * space).values,
            name=f'{i+1} shift',
            orientation='h',
            insidetextanchor='start',
            marker=dict(
                color=colors[color_id],
                line=dict(color='beige', width=13)
            ),
            showlegend=False
        ))
        
        lags_ser = df['lags_list'].apply(lambda x: find_shift_n(x, n=i))
        row = 0

        for idx in lags_ser.index:
            shifts_lag = lags_ser[idx]
            ann_size = pos_series.loc[idx]
            for i in range(len(shifts_lag)):
                annotations.extend([
                    dict(x=ann_size, y=row, text=shifts_lag[i].strip('()'), 
                            font=dict(family='Lucida Console', size=ann_font_size, color='beige'),
                            showarrow=False) #, bgcolor="darkkhaki"
                    ])
                ann_size += 40
            row += 1
            pos_series.loc[idx] = ann_size

    height_bin = 70
    if len(df) < 3:
        height_bin = 250
    elif  3 <= len(df) <= 5:
         height_bin = 100
    fig.update_layout(
        barmode='stack', annotations=annotations,
        width=1000,
        height=len(df) * height_bin,
        xaxis=dict(mirror=True)
    )
    fig.update_yaxes(
        tickvals = [i for i in range(len(df))], 
        ticktext=custom_ticktext, 
        tickfont=dict(family='Lucida Console', size=18),
        side="right")
    fig.update_xaxes(visible=False)
    return fig

def plot_airport_pairing(data_with_shifts, airport):
    mask = data_with_shifts['airports'].apply(lambda x: airport == x[0])
    fig3 = plot_pairings(data_with_shifts[mask])
    st.text(f'Pairings with the base station {airport}')
    st.plotly_chart(fig3, use_container_width=True)


def add_none_option(options: pd.Series or set):
    return ['None'] + list(options)

# Main
st.set_page_config(
    page_title="ComPairings App",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://docs.streamlit.io/library/api-reference',
        # 'Report a bug': "https://",
        'About': "## This is an app for pairings comparison!"
    }
)
st.title('ComPare Pairings')

left_column, right_column = st.columns(2)
with left_column:
    # Create a file uploader component
    uploaded_file1 = st.file_uploader("Choose .txt File#1")

    # Check if a file was uploaded
    if uploaded_file1 is not None:
        data1 = read_data(uploaded_file1)
    else:
        st.warning("You didn't choose File#1", icon="⚠️")
        sys.exit(0)

    with_shift_and_lags1 = extract_shift_and_lag(data1)
    
with right_column:

    uploaded_file2 = st.file_uploader("Choose .txt File#2")

    if uploaded_file2 is not None:
        data2 = read_data(uploaded_file2)
    else:
        st.warning("You didn't choose File#2", icon="⚠️")
        sys.exit(0)

    with_shift_and_lags2 = extract_shift_and_lag(data2)
    

if st.sidebar.button('Show all pairings'):
        with left_column:
            st.text("All pairings from File#1")
            fig1 = plot_pairings(with_shift_and_lags1)
            st.plotly_chart(fig1, use_container_width=True)
        with right_column:
            st.text("All pairings from File#2")
            fig2 = plot_pairings(with_shift_and_lags2)
            st.plotly_chart(fig2, use_container_width=True)

with left_column:
    pairing_to_filter1 = st.slider('Show N pairings from File#1', 0, len(with_shift_and_lags1), 0)
    
    if pairing_to_filter1 != 0:

        df1 = with_shift_and_lags1.copy().head(pairing_to_filter1)
        fig1 = plot_pairings(df1)
        st.plotly_chart(fig1, use_container_width=True)

with right_column:
    pairing_to_filter2 = st.slider('Show N pairings from File#2', 0, len(with_shift_and_lags2), 0)

    if pairing_to_filter2 != 0:

        df2 = with_shift_and_lags2.copy().head(pairing_to_filter2)
        fig2 = plot_pairings(df2)
        st.plotly_chart(fig2, use_container_width=True)
        


# Plot pairings with chosen airport
with left_column:
    with_shift_and_lags1['airports'] = with_shift_and_lags1['pairing'].apply(lambda x: list_airports(x))
    print(with_shift_and_lags1)
    base_stations1 = np.unique(with_shift_and_lags1['airports'].apply(lambda x: x[0]).values)
    print(base_stations1)
    airport1 = st.sidebar.selectbox(f'Choose base station 1', add_none_option(base_stations1))
    if airport1 != 'None':
        plot_airport_pairing(with_shift_and_lags1, airport=airport1)

with right_column:
    with_shift_and_lags2['airports'] = with_shift_and_lags2['pairing'].apply(lambda x: list_airports(x))
    base_stations2 = np.unique(with_shift_and_lags2['airports'].apply(lambda x: x[0]).values)
    airport2 = st.sidebar.selectbox(f'Choose base station 2', add_none_option(base_stations2))
    if airport2 != 'None':
        plot_airport_pairing(with_shift_and_lags2, airport=airport2)


# Pairings comparison one by one
with left_column:
    option1 = st.sidebar.selectbox(
        'Choose pairing 1',
        add_none_option(with_shift_and_lags1['pairing']))
    if option1 != "None":
        st.subheader(f'First:')
        st.text(option1)
        df1 = with_shift_and_lags1.query('pairing == @option1')
        fig1 = plot_pairings(df1)
        st.plotly_chart(fig1, use_container_width=True)

with right_column:
    option2 = st.sidebar.selectbox(
       'Choose pairing 2',
        add_none_option(with_shift_and_lags2['pairing']))
    if option2 != "None":
        st.subheader(f'Second:')
        st.text(option2)
        df2 = with_shift_and_lags2.query('pairing == @option2')
        fig2 = plot_pairings(df2)
        st.plotly_chart(fig2, use_container_width=True)


# Show raw data
if st.sidebar.checkbox('Show raw data 1'):
    with left_column:
        st.subheader('Raw data (File 1)')
        st.write(with_shift_and_lags1[['pairing', 'number']])

if st.sidebar.checkbox('Show raw data 2'):
    with right_column:
        st.subheader('Raw data (File 2)')
        st.write(with_shift_and_lags2[['pairing', 'number']])
