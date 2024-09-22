import streamlit as st
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

# Load data and preprocess
@st.cache_data()
def load_data():
    movies_data = pd.read_csv('data/movies.csv')
    selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
    for feature in selected_features:
        movies_data[feature] = movies_data[feature].fillna('')
    return movies_data

movies_data = load_data()

# Function to combine features for recommendations
@st.cache_data()
def get_combined_features():
    return movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']

# Function to recommend similar movies
@st.cache_data()
def recommend_similar_movies(selected_movie, num_recommendations=20):
    combined_feature = get_combined_features()
    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(combined_feature)
    similarity = cosine_similarity(feature_vectors)
    find_close_match = difflib.get_close_matches(selected_movie, movies_data['title'].tolist())
    if find_close_match:
        close_match = find_close_match[0]
        index_of_the_movie = movies_data[movies_data['title'] == close_match].index[0]
        similarity_score = list(enumerate(similarity[index_of_the_movie]))
        sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
        similar_movies = []
        for i, movie in enumerate(sorted_similar_movies[:num_recommendations]):
            index = movie[0]
            title_from_index = movies_data.loc[index, 'title']
            similar_movies.append(title_from_index)
        return similar_movies
    return []

# Function to fetch movie poster using TMDb API
@st.cache_data()
def fetch_movie_poster(movie_title):
    api_key = "ffc42ca6e427cdb9fa3a287ab41c970d"
    url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={movie_title}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if data['results']:
            poster_path = data['results'][0]['poster_path']
            if poster_path:
                poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
                return poster_url
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data from TMDb: {e}")
    return None

# Sidebar - Movie Selection
st.sidebar.title('List of Movies')
selected_movie = st.sidebar.selectbox('Select a movie:', movies_data['title'].tolist())
num_recommendations = st.sidebar.slider('Number of recommendations:', min_value=1, max_value=50, value=10)

if selected_movie:
    # Recommended Movies
    st.title(f"Movies similar to '{selected_movie}':")
    
    with st.spinner('Fetching recommendations...'):
        similar_movies = recommend_similar_movies(selected_movie, num_recommendations)
    
    for i, movie in enumerate(similar_movies):
        movie_poster = fetch_movie_poster(movie)
        if movie_poster:
            st.image(movie_poster, caption=movie, use_column_width=True)
        else:
            st.write(f"{i + 1}. {movie}")
