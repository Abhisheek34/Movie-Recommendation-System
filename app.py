import streamlit as st
import pickle
import difflib
from sklearn.metrics.pairwise import cosine_similarity

# Load pickled components
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('titles.pkl', 'rb') as f:
    list_titles = pickle.load(f)

# Load feature vector and compute similarity
with open('feature_vector.pkl', 'rb') as f:
    feature_vector = pickle.load(f)

similarity = cosine_similarity(feature_vector)

# Streamlit UI
st.title("🎬 Movie Recommendation System")
movie_name = st.text_input("Enter your favourite movie name:")

if movie_name:
    find_close_match = difflib.get_close_matches(movie_name, list_titles)
    
    if find_close_match:
        close_match = find_close_match[0]
        movie_index = list_titles.index(close_match)
        similarity_score = list(enumerate(similarity[movie_index]))
        sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

        st.subheader(f"Movies suggested for you based on '{close_match}':")
        for i, movie in enumerate(sorted_similar_movies[1:11], start=1):
            index = movie[0]
            recommended_title = list_titles[index]
            st.write(f"{i}. {recommended_title}")
    else:
        st.warning("No close match found. Please try another movie name.")
