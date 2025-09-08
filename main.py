from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import difflib
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- Load Pickled Models ----------------
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('titles.pkl', 'rb') as f:
    list_titles = pickle.load(f)

with open('feature_vector.pkl', 'rb') as f:
    feature_vector = pickle.load(f)

similarity = cosine_similarity(feature_vector)

# ---------------- FastAPI App ----------------
app = FastAPI(title="Movie Recommendation API")

# Input schema
class MovieRequest(BaseModel):
    movie_name: str

@app.post("/recommend")
def recommend_movies(request: MovieRequest):
    movie_name = request.movie_name

    # Find closest match
    find_close_match = difflib.get_close_matches(movie_name, list_titles)
    
    if not find_close_match:
        raise HTTPException(status_code=404, detail="Movie not found. Try another title.")

    close_match = find_close_match[0]
    movie_index = list_titles.index(close_match)
    similarity_score = list(enumerate(similarity[movie_index]))
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    recommendations = []
    for i, movie in enumerate(sorted_similar_movies[1:11], start=1):
        index = movie[0]
        recommended_title = list_titles[index]
        recommendations.append(recommended_title)

    return {"input_movie": close_match, "recommendations": recommendations}