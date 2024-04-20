import pandas as pd
import numpy as np
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk

# Load movie data from CSV file
df = pd.read_csv(r"movies.csv")

# Function to load and prepare movie data
def get_movie_data():

    # Select relevant features for recommendation
    selected_features = ["genres", "keywords", "tagline", "cast", "director"]

    # Replace null values with empty strings
    for feature in selected_features:
        df[feature] = df[feature].fillna("")

    # Combine selected features
    combined_features = (
        df["genres"]
        + " "
        + df["keywords"]
        + " "
        + df["tagline"]
        + " "
        + df["cast"]
        + " "
        + df["director"]
    )

    return df, combined_features

# Function to calculate cosine similarity and recommend movies
def get_movie_recommendations(combined_features):
    # Convert text data to feature vectors
    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(combined_features)

    # Calculate cosine similarity between movies
    similarity = cosine_similarity(feature_vectors)

    # Function to recommend movies based on user input
    def recommend_movies(movie_name):
        # Find the closest match for the user input
        list_of_all_titles = df["title"].tolist()
        find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
        close_match = find_close_match[0]

        # Get index of the movie
        index_of_the_movie = df[df.title == close_match]["index"].values[0]

        # Get similarity scores and sort them
        similarity_score = list(enumerate(similarity[index_of_the_movie]))
        sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

        # Return top 10 recommendations
        return sorted_similar_movies[:10]

    return recommend_movies

# Function to create and run the GUI
def main():
    # Load movie data and get recommendation function
    df, combined_features = get_movie_data()
    recommend_movies = get_movie_recommendations(combined_features)

    # Create the main window
    root = tk.Tk()
    root.title("Movie Recommender")

    # Label and input field for movie name
    movie_label = tk.Label(root, text="Enter your favorite movie name :")
    movie_label.pack()

    movie_entry = tk.Entry(root)
    movie_entry.pack()

    # Button to trigger recommendation
    recommend_button = tk.Button(
        root, text="Recommend Movies", command=lambda: display_recommendations(movie_entry.get())
    )
    recommend_button.pack()

    # Listbox to display recommendations
    recommendation_list = tk.Listbox(root, width=50, height=15)
    recommendation_list.pack()

    # Function to display recommendations in the listbox
    def display_recommendations(movie_name):
        recommendations = recommend_movies(movie_name)

        recommendation_list.delete(0, tk.END)
        for i, (index, score) in enumerate(recommendations):
            title = df[df.index == index]["title"].values[0]
            recommendation_list.insert(tk.END, f"{i + 1}. {title} (Score: {score:.2f})")

    # Start the main loop
    root.mainloop()

# Run the main function
if __name__ == "__main__":
    main()
