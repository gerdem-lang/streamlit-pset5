import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="Simple Anime Explorer", layout="wide")
st.title("Simple Anime Explorer (Beginner)")

st.markdown(
    "Upload the Kaggle anime.csv file (from the anime-recommendations-database) or try the small sample dataset below."
)

# Allow user to upload a CSV. If none uploaded, use a tiny built-in sample so the app always runs.
uploaded = st.file_uploader("Upload anime.csv (optional)", type=["csv"])

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()
else:
    # Tiny sample so beginners can see the app working without any files
    sample = {
        "anime_id": [1, 2, 3, 4, 5],
        "name": [
            "Fullmetal Alchemist: Brotherhood",
            "Kimi no Na wa.",
            "Death Note",
            "Some Slice of Life",
            "Action Show"
        ],
        "genre": [
            "Action, Adventure, Fantasy",
            "Romance, Supernatural, Drama",
            "Mystery, Thriller, Supernatural",
            "Slice of Life, Comedy",
            "Action, Adventure"
        ],
        "type": ["TV", "Movie", "TV", "TV", "TV"],
        "episodes": [64, 1, 37, 12, 24],
        "rating": [9.25, 8.9, 8.6, 7.0, 7.8],
        "members": [2000000, 1500000, 1800000, 50000, 120000]
    }
    df = pd.DataFrame(sample)
    st.info("No file uploaded — using a tiny built-in sample dataset for demonstration.")

# Basic cleaning: ensure expected columns
expected = {"anime_id", "name", "genre", "type", "episodes", "rating", "members"}
present = expected.intersection(set(df.columns))
if "name" not in df.columns:
    st.error("The CSV must contain a 'name' column. If using Kaggle's dataset, upload 'anime.csv'.")
    st.stop()

# Normalize genres into list
df["genre"] = df["genre"].fillna("")
df["genres_list"] = df["genre"].apply(lambda s: [g.strip() for g in s.split(",") if g.strip()])

# Sidebar controls
st.sidebar.header("Controls")
all_genres = sorted({g for gl in df["genres_list"] for g in gl})
genre_choice = st.sidebar.selectbox("Genre", options=["All"] + all_genres)
top_n = st.sidebar.slider("Top N (by members)", min_value=1, max_value=20, value=5)
search = st.sidebar.text_input("Search title (partial)")

# Apply filters
filtered = df.copy()
if genre_choice != "All":
    filtered = filtered[filtered["genres_list"].apply(lambda gl: genre_choice in gl)]

if search:
    filtered = filtered[filtered["name"].str.contains(search, case=False, na=False)]

# Show top N by members
st.subheader(f"Top {top_n} anime by members (filtered)")
if "members" in filtered.columns:
    top = filtered.sort_values("members", ascending=False).head(top_n)
    chart = (
        alt.Chart(top)
        .mark_bar()
        .encode(
            x=alt.X("members:Q", title="Members (popularity)"),
            y=alt.Y("name:N", sort=alt.EncodingSortField(field="members", order="descending")),
            tooltip=["name", "type", "episodes", "rating", "members"],
        )
        .properties(height=40 * len(top))
    )
    st.altair_chart(chart, use_container_width=True)
    st.write(top[["anime_id", "name", "type", "episodes", "rating", "members"]])
else:
    st.write("No 'members' column available in the uploaded CSV. Showing available rows:")
    st.dataframe(filtered.head(top_n))

# Optional: rating histogram for a selected anime
st.subheader("Inspect an anime")
selected = st.selectbox("Choose an anime to see details", options=filtered["name"].tolist() + ["None"])
if selected and selected != "None":
    row = df[df["name"] == selected].iloc[0]
    st.write(row[["anime_id", "name", "type", "episodes", "rating", "members", "genre"]])

st.caption("If you want help deploying this to Streamlit Cloud or pushing to GitHub, tell me your GitHub username and I will give exact commands.")