import streamlit as st
from datetime import datetime

today = datetime.today().strftime("%H:%M:%S")

st.title(today)

model = st.selectbox(
    "Choose your LLM",
    (
        "GPT-3.5",
        "GPT-4",
    ),
)

if model == "GPT-3.5":
    st.write("cheap")
else:
    st.write("not cheap")
    name = st.text_input("What is your name?")
    st.write(name)

    value = st.slider(
        "templature",
        min_value=0.0,
        max_value=1.0,
    )

    st.write(value)
