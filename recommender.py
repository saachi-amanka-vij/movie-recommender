import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv("data/movies.csv")

# Preprocess
movies["genre"] = movies["genre"].fillna("")
movies["overview"] = movies["overview"].fillna("")

movies["combined_features"] = (
    movies["genre"] + " " + movies["overview"]
).str.lower()

# TF-IDF
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["combined_features"])

# Similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Mapping
movies["title_lower"] = movies["title"].str.lower()
title_to_index = pd.Series(movies.index, index=movies["title_lower"])

# Recommendation function
def recommend(movie_name, n=5):
    movie_name = movie_name.lower()

    if movie_name not in title_to_index:
        return ["Movie not found"]

    idx = title_to_index[movie_name]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]

    return [
        f"{movies['title'].iloc[i]} (score: {round(score, 2)})"
        for i, score in sim_scores
    ]