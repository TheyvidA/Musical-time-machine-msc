import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

# 1. Page Configuration
st.set_page_config(page_title="The Musical Time Machine", layout="centered")
st.title("🕰️ The Musical Time Machine")
st.write("Find the historical sonic twin of any modern track.")


# 2. Load and Cache Data (Crucial for the 3.0s latency target!)
@st.cache_data
def load_data():
    df = pd.read_csv('spotify_master_cleaned.csv')
    return df


df = load_data()
features = ['acousticness', 'danceability', 'energy', 'loudness', 'tempo', 'valence']

# Scale the features
scaler = MinMaxScaler()
df_scaled = df.copy()
df_scaled[features] = scaler.fit_transform(df[features])

# 3. User Interface Controls
st.markdown("### Step 1: Initialize the Machine")
source_song = st.text_input("Enter a modern song you like:", placeholder="e.g., Blinding Lights")

target_decade = st.selectbox(
    "Select your target destination:",
    [1970, 1980, 1990, 2000, 2010],
    format_func=lambda x: f"The {x}s"
)

# 4. The k-NN Engine
if st.button("🚀 Travel Through Time"):
    if source_song:
        with st.spinner("Calculating multidimensional acoustic vectors..."):

            # Find the song
            match = df_scaled[df_scaled['track_name'].str.contains(source_song, case=False, regex=False)]

            if match.empty:
                st.error("Song not found in the database. Try another track!")
            else:
                source_vector = match[features].iloc[0].values.reshape(1, -1)
                source_info = df[df['track_name'].str.contains(source_song, case=False, regex=False)].iloc[0]

                # Filter to decade
                vintage_pool = df_scaled[
                    (df_scaled['year'] >= target_decade) & (df_scaled['year'] < target_decade + 10)]

                if not vintage_pool.empty:
                    # Run k-NN
                    knn = NearestNeighbors(n_neighbors=4, metric='euclidean')
                    knn.fit(vintage_pool[features])
                    distances, indices = knn.kneighbors(source_vector)

                    # Display Results
                    st.success("Time Jump Successful!")
                    st.markdown(
                        f"**Source Track:** *{source_info['track_name']}* by {source_info['artists_name']} ({int(source_info['year'])})")
                    st.divider()

                    st.subheader(f"Top Matches from the {target_decade}s:")

                    for i in range(1, 4):
                        match_idx = vintage_pool.index[indices[0][i]]
                        matched_song = df.loc[match_idx]
                        dist_score = round(distances[0][i], 4)

                        st.info(
                            f"🎵 **{matched_song['track_name']}** by {matched_song['artists_name']} ({int(matched_song['year'])})  \n*Distance Score: {dist_score}*")
