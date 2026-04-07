import streamlit as st
from recommender import recommend

st.title("🎬 Movie Recommender")

movie = st.text_input("Enter a movie")

if st.button("Recommend"):
    results = recommend(movie)

    for r in results:
        st.write(r)