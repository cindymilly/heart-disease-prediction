import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
st.button("Test", use_container_width=True)
st.dataframe(pd.DataFrame({'A': [1]}), use_container_width=True)
fig, ax = plt.subplots()
st.pyplot(fig, use_container_width=True)
