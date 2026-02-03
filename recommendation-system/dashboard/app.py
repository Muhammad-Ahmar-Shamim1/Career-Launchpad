import streamlit as st
import requests

st.set_page_config(layout="wide")

st.title("🎬 Movie Recommendation Dashboard")

# Sidebar controls
st.sidebar.header("Settings")
user_id = st.sidebar.selectbox("Select User ID", range(1, 101))
num_recommendations = st.sidebar.slider("Number of Recommendations", min_value=1, max_value=20, value=5, step=1)

if st.button("Get Recommendations"):
    res = requests.get(
        "http://127.0.0.1:8000/recommend",
        params={"user_id": user_id, "top_n": num_recommendations}
    ).json()

    st.subheader(f"⭐ Top {num_recommendations} Picks for User {user_id}")
    
    # Display recommendations in grid
    cols = st.columns(min(5, num_recommendations))
    
    for idx, movie in enumerate(res["recommendations"]):
        col = cols[idx % 5]
        with col:
            st.markdown(f"### {movie['title'][:25]}..." if len(movie['title']) > 25 else f"### {movie['title']}")
            st.markdown(f"**Movie ID:** {movie['movieId']}")
            st.info(f"Recommended based on similar users' preferences")
    
    # Show summary
    st.success(f"✅ Showing {len(res['recommendations'])} recommendations for User {user_id}")
