import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from rapidfuzz import process

movies = pd.read_csv("data/movies.csv")

movies["genre"] = movies["genre"].fillna("")
movies["overview"] = movies["overview"].fillna("")

movies["combined_features"] = (movies["genre"] * 2 + " " + movies["overview"]).str.lower()

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["combined_features"])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

kmeans = KMeans(n_clusters=4, random_state=42)
movies["cluster"] = kmeans.fit_predict(tfidf_matrix)

movies["title_lower"] = movies["title"].str.lower()
title_to_index = pd.Series(movies.index, index=movies["title_lower"])

def get_closest_title(name):
    return process.extractOne(name, movies["title"])[0]

def get_top_keywords(idx, top_n=5):
    feature_array = tfidf.get_feature_names_out()
    tfidf_scores = tfidf_matrix[idx].toarray().flatten()
    top_indices = tfidf_scores.argsort()[-top_n:][::-1]
    return ", ".join(feature_array[i] for i in top_indices)

def recommend(movie_name, n=5):
    movie_name = movie_name.lower()
    if movie_name not in title_to_index:
        movie_name = get_closest_title(movie_name).lower()

    idx = title_to_index[movie_name]
    sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)[1:n+1]

    return [{
        "title": movies["title"].iloc[i],
        "score": round(score, 2),
        "reason": get_top_keywords(i)
    } for i, score in sim_scores]

def recommend_by_context(mood, weather, n=5):
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

    filtered = movies[movies["overview"].str.contains("|".join(keywords), case=False)]

    return filtered["title"].head(n).tolist()

def recommend_from_cluster(movie_name, n=5):
    movie_name = movie_name.lower()
    if movie_name not in title_to_index:
        movie_name = get_closest_title(movie_name).lower()

    idx = title_to_index[movie_name]
    cluster_id = movies["cluster"].iloc[idx]

    return movies[movies["cluster"] == cluster_id]["title"].head(n).tolist()