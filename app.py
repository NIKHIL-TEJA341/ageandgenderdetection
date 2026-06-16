import streamlit as st
import app1

st.set_page_config(
    page_title="Age and Gender Detection",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

from tasks import (
    task1_longhair,
    task2_seniorcitizen,
    task4_signlanguage,
    task5_carcolour,
)

def main():
    st.sidebar.title("Navigation")
    st.sidebar.markdown("Select a task to view:")
    
    app_mode = st.sidebar.radio(
        "Tasks",
        [
            "Age and Gender Detection",
            "Long Hair Identification",
            "Senior Citizen Detection",
            "Sign Language Detection",
            "Car Colour & Counting"
        ]
    )

    if app_mode == "Age and Gender Detection":
        app1.main()
    elif app_mode == "Long Hair Identification":
        task1_longhair.show_page()
    elif app_mode == "Senior Citizen Detection":
        task2_seniorcitizen.show_page()
    elif app_mode == "Sign Language Detection":
        task4_signlanguage.show_page()
    elif app_mode == "Car Colour & Counting":
        task5_carcolour.show_page()

if __name__ == "__main__":
    main()
