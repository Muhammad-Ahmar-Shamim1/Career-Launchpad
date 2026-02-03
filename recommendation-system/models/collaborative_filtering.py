import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class CollaborativeFiltering:
    def __init__(self, ratings_path):
        self.ratings = pd.read_csv(ratings_path)
        self.user_item = None
        self.similarity = None

    def train(self):
        self.user_item = self.ratings.pivot(
            index='userId',
            columns='movieId',
            values='rating'
        ).fillna(0)

        self.similarity = cosine_similarity(self.user_item)

    def recommend(self, user_id, top_n=15):
        user_idx = user_id - 1
        user_rated = self.user_item.iloc[user_idx]
        
        # Get weighted scores from similar users
        scores = self.similarity[user_idx]
        similar_users = scores.argsort()[::-1][1:min(31, len(self.user_item))]
        
        # Get ratings from similar users
        similar_ratings = self.user_item.iloc[similar_users]
        
        # Calculate weighted average based on similarity scores
        weights = scores[similar_users].reshape(-1, 1)
        weighted_ratings = (similar_ratings * weights).sum(axis=0) / weights.sum()
        
        # Filter out movies already rated by the user
        unrated_movies = weighted_ratings[user_rated == 0]
        recommendations = unrated_movies.sort_values(ascending=False)
        
        return recommendations.head(top_n).index.tolist()
