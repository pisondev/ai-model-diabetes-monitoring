import streamlit as st

from .config import APP_TITLE, DISCLAIMER


def page(title, subtitle=None):
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def source_badge(source):
    if source == "processed":
        st.success("Data source: frozen milestone dataset")
    else:
        st.warning("Data source: generated dummy data, no real record is shown")


def disclaimer():
    st.caption(DISCLAIMER)
