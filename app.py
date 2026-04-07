import streamlit as st
from recommender import (
    recommend,
    movies,
    recommend_by_context,
    recommend_from_cluster
)

# Page config
st.set_page_config(page_title="Movie Recommender", layout="wide")

# Header
st.markdown(
    """
    <div style='text-align:center; padding:20px; background: linear-gradient(90deg,#1f4037,#99f2c8); border-radius:15px;'>
        <h1 style='color:white;'>🎬 Movie Recommender</h1>
        <p style='color:white;'>Smart recommendations using ML, Mood & Weather</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")

movie_list = movies["title"].tolist()

# ------------------ MAIN RECOMMENDER ------------------
st.markdown("## 🔍 Find Similar Movies")

col1, col2 = st.columns([3,1])

with col1:
    selected_movie = st.selectbox("Choose a movie", movie_list)

with col2:
    st.write("")
    st.write("")
    recommend_btn = st.button("✨ Recommend")

if recommend_btn:
    results = recommend(selected_movie)

    st.markdown("### 🎯 Recommendations")

    cols = st.columns(3)

    for idx, movie in enumerate(results):
        with cols[idx % 3]:
            st.markdown(
                f"""
                <div style="background:#1e1e1e;padding:15px;border-radius:12px;margin-bottom:15px;">
                    <h4 style="color:white;">🎥 {movie['title']}</h4>
                    <p style="color:#00ffcc;">⭐ {movie['score']}</p>
                    <p style="color:#cccccc;">💡 {movie['reason']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

# ------------------ MOOD + WEATHER ------------------
st.markdown("---")
st.markdown("## 🌦🎭 Mood + Weather Based Recommendations")

col1, col2, col3 = st.columns(3)

with col1:
    mood = st.selectbox("🎭 Mood", ["Feel Good 😊", "Emotional 😢", "Mind Bending 🤯", "Action 🔥"])

with col2:
    weather = st.selectbox("🌦 Weather", ["Sunny ☀️", "Rainy 🌧", "Stormy ⛈", "Hot 🔥"])

with col3:
    st.write("")
    st.write("")
    context_btn = st.button("🌍 Recommend")

if context_btn:
    mood_key = mood.split()[0].lower()
    weather_key = weather.split()[0].lower()

    results = recommend_by_context(mood_key, weather_key)

    st.markdown("### 🎬 Personalized Picks")

    for movie in results:
        st.write(f"🎬 {movie}")

# ------------------ CLUSTER ------------------
st.markdown("---")
st.markdown("## 🧠 Explore Similar Movie Clusters")

cluster_movie = st.selectbox("Pick a movie", movie_list)

if st.button("🔍 Explore Cluster"):
    cluster_results = recommend_from_cluster(cluster_movie)

    for movie in cluster_results:
        st.write(f"🎬 {movie}")

# Footer
st.markdown("---")
st.markdown("<p style='text-align:center;color:gray;'>Built with ❤️ using ML</p>", unsafe_allow_html=True)