from fastapi import FastAPI
import pandas as pd
import sys
import os
sys.path.insert(0, "..")
from models.collaborative_filtering import CollaborativeFiltering

app = FastAPI()

# Get the parent directory for data files
data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

cf = CollaborativeFiltering(os.path.join(data_dir, "ratings.csv"))
cf.train()

movies = pd.read_csv(os.path.join(data_dir, "movies.csv"))

@app.get("/recommend")
def recommend(user_id: int, top_n: int = 5):
    movie_ids = cf.recommend(user_id, top_n=top_n)

    result = movies[movies['movieId'].isin(movie_ids)][
        ['movieId', 'title']
    ]

    return {
        "user_id": user_id,
        "recommendations": result.to_dict(orient="records"),
        "count": len(result)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
