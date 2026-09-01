import streamlit as st

from .config import APP_TITLE, DISCLAIMER

SCROLLBAR_THUMB = "#cbd5e1"


def light_mode_css():
    """Covers the chrome the browser paints itself, which the theme in config.toml cannot reach."""
    surface = st.get_option("theme.secondaryBackgroundColor")
    return "\n".join(
        [
            "<style>",
            # streamlit leaves this at normal, so a dark system scheme still darkens scrollbars and native controls
            "  :root, html, body, .stApp { color-scheme: light; }",
            f"  * {{ scrollbar-color: {SCROLLBAR_THUMB} {surface}; }}",
            "</style>",
        ]
    )


def page(title, subtitle=None):
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.markdown(light_mode_css(), unsafe_allow_html=True)
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def figure(title, chart, note=None, container_width=True):
    st.markdown(f"**{title}**")
    st.altair_chart(chart, use_container_width=container_width)
    if note:
        st.caption(note)


def source_badge(source):
    if source == "processed":
        st.success("Data source: frozen milestone dataset")
    else:
        st.warning("Data source: generated dummy data, no real record is shown")


def disclaimer():
    st.caption(DISCLAIMER)
