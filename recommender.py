import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from rapidfuzz import process

# ------------------ LOAD DATA ------------------
movies = pd.read_csv("data/movies.csv")

# ------------------ PREPROCESS ------------------
movies["genre"] = movies["genre"].fillna("")
movies["overview"] = movies["overview"].fillna("")

# Weighted features (better ML)
movies["combined_features"] = (
    movies["genre"] * 2 + " " + movies["overview"]
).str.lower()

# ------------------ TF-IDF ------------------
tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf_matrix = tfidf.fit_transform(movies["combined_features"])

# ------------------ SIMILARITY ------------------
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Normalize
cosine_sim = (cosine_sim - cosine_sim.min()) / (cosine_sim.max() - cosine_sim.min())

# ------------------ CLUSTERING ------------------
kmeans = KMeans(n_clusters=4, random_state=42)
movies["cluster"] = kmeans.fit_predict(tfidf_matrix)

# ------------------ MAPPING ------------------
movies["title_lower"] = movies["title"].str.lower()
title_to_index = pd.Series(movies.index, index=movies["title_lower"])

# ------------------ FUZZY MATCH ------------------
def get_closest_title(name):
    match = process.extractOne(name, movies["title"])
    return match[0]

# ------------------ KEYWORDS ------------------
def get_top_keywords(idx, top_n=5):
    feature_array = tfidf.get_feature_names_out()
    tfidf_scores = tfidf_matrix[idx].toarray().flatten()
    top_indices = tfidf_scores.argsort()[-top_n:][::-1]
    return ", ".join(feature_array[i] for i in top_indices)

# ------------------ MAIN RECOMMENDER ------------------
def recommend(movie_name, n=5):
    movie_name = movie_name.lower()

    if movie_name not in title_to_index:
        movie_name = get_closest_title(movie_name).lower()

    idx = title_to_index[movie_name]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]

    results = []
    for i, score in sim_scores:
        results.append({
            "title": movies["title"].iloc[i],
            "score": round(float(score), 2),
            "reason": get_top_keywords(i)
        })

    return results

# ------------------ MOOD + WEATHER ------------------
def recommend_by_context(mood, weather, n=5):
    mood = mood.lower()
    weather = weather.lower()

    mood_map = {
        "feel": ["fun", "happy"],
        "emotional": ["life", "family"],
        "mind": ["dream", "time"],
        "action": ["war", "battle"]
    }

    weather_map = {
        "sunny": ["adventure"],
        "rainy": ["love"],
        "stormy": ["dark"],
        "hot": ["action"]
    }

    keywords = mood_map.get(mood.split()[0], []) + weather_map.get(weather, [])

    if not keywords:
        return []

    filtered = movies[
        movies["overview"].str.contains("|".join(keywords), case=False)
    ]

    return filtered["title"].head(n).tolist()

# ------------------ CLUSTER RECOMMENDER ------------------
def recommend_from_cluster(movie_name, n=5):
    movie_name = movie_name.lower()

    if movie_name not in title_to_index:
        movie_name = get_closest_title(movie_name).lower()

    idx = title_to_index[movie_name]
    cluster_id = movies["cluster"].iloc[idx]

    similar_movies = movies[movies["cluster"] == cluster_id]

    return similar_movies["title"].head(n).tolist()