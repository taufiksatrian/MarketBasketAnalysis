import streamlit as st
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import association_rules, apriori

data = pd.read_csv("bread basket.csv", low_memory=False, sep=',')
data['date_time'] = pd.to_datetime(data['date_time'], format='%d-%m-%Y %H:%M')

data['month'] = data['date_time'].dt.month
data['day'] = data['date_time'].dt.weekday

month_mapping = {
    1: 'January',
    2: 'February',
    3: 'March',
    4: 'April',
    5: 'May',
    6: 'June',
    7: 'July',
    8: 'August',
    9: 'September',
    10: 'October',
    11: 'November',
    12: 'December'
}
day_mapping = {
    0: 'Monday',
    1: 'Tuesday',
    2: 'Wednesday',
    3: 'Thursday',
    4: 'Friday',
    5: 'Saturday',
    6: 'Sunday'
}

data['month'] = data['month'].map(month_mapping)
data['day'] = data['day'].map(day_mapping)

st.title("Market Basket Analysis Menggunakan Algoritma Apriori")


def get_data(period_day='', weekday_weekend='', month='', day=''):
    data_process = data.copy()
    filtered = data_process.loc[
        (data_process["period_day"].str.contains(period_day)) &
        (data_process["weekday_weekend"].str.contains(weekday_weekend)) &
        (data_process["month"].str.contains(month.title())) &
        (data_process["day"].str.contains(day.title()))
    ]
    return filtered if not filtered.empty else "No Result!"


def user_input_features():
    item = st.selectbox("Item", data["Item"].unique())
    period_day = st.selectbox(
        "Period Day", ['Morning', 'Afternoon', 'Evening', 'Night'])
    weekday_weekend = st.selectbox("Weekday / Weekend", ['Weekday', 'Weekend'])
    month = st.select_slider("Month", [
                             'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    day = st.select_slider(
        "Day", ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], value='Sat')

    return period_day, weekday_weekend, month, day, item


period_day, weekday_weekend, month, day, item = user_input_features()
data_process = get_data(
    period_day.lower(), weekday_weekend.lower(), month, day)


def encode(x):
    if x <= 0:
        return 0
    elif x >= 1:
        return 1


if type(data_process) != type("No Result!"):
    item_count = data_process.groupby(["Transaction", "Item"])[
        "Item"].count().reset_index(name="Count")
    item_count_pivot = item_count.pivot_table(
        index='Transaction', columns='Item', values='Count', aggfunc='sum').fillna(0)
    item_count_pivot = item_count_pivot.applymap(encode)

    support = 0.01
    frequent_items = apriori(
        item_count_pivot, min_support=support, use_colnames=True)

    metric = 'lift'
    min_threshold = 1

    rules = association_rules(
        frequent_items, metric=metric, min_threshold=min_threshold)[["antecedents", "consequents", "support", "confidence", "lift"]]
    rules.sort_values('confidence', ascending=False, inplace=True)


def parse_list(x):
    x = list(x)
    if len(x) == 1:
        return x[0]
    elif len(x) > 1:
        return ", ".join(x)


def return_item_df(item_antecedents):
    data_process = rules[["antecedents", "consequents"]].copy()
    data_process["antecedents"] = data_process["antecedents"].apply(parse_list)
    data_process["consequents"] = data_process["consequents"].apply(parse_list)
    matching_rows = data_process.loc[data_process['antecedents']
                                     == item_antecedents]
    if not matching_rows.empty:
        return list(matching_rows.iloc[0, :])
    else:
        return "No matching data found"


if type(data_process) != type("No Result!"):
    st.markdown("Hasil Rekomendasi : ")
    st.success(
        f"Jika Konsumen Membeli **{ item }**, Maka Membeli **{ return_item_df(item)[1] }** Secara Bersamaan")
