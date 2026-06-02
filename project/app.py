import streamlit as st
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

# 1. Page Configuration
st.set_page_config(page_title="The Musical Time Machine", layout="centered")
st.title("🕰️ The Musical Time Machine")
st.write("Find the historical sonic twin of any modern track.")


# 2. Load and Cache Data (Crucial for the 3.0s latency target!)
@st.cache_data
def load_data():
    # Get the exact directory where app.py is located
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Build the path to the CSV file dynamically
    csv_path = os.path.join(current_dir, 'spotify_master_cleaned.csv')

    # Load the data
    df = pd.read_csv(csv_path)
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
                    # Run k-NN (Get top 4 just in case we need to skip a duplicate)
                    knn = NearestNeighbors(n_neighbors=4, metric='euclidean')
                    knn.fit(vintage_pool[features])
                    distances, indices = knn.kneighbors(source_vector)
                    
                    # Display Results
                    st.success("Time Jump Successful!")
                    st.markdown(f"**Source Track:** *{source_info['track_name']}* by {source_info['artists_name']} ({int(source_info['year'])})")
                    st.divider()
                    
                    st.subheader(f"Top Matches from the {target_decade}s:")
                    
                    # Track how many valid matches we've shown
                    matches_shown = 0 
                    
                    # Loop through the results, starting at 0 (the absolute closest)
                    for i in range(len(indices[0])):
                        if matches_shown >= 3:
                            break # Stop once we have 3 good recommendations
                            
                        match_idx = vintage_pool.index[indices[0][i]]
                        matched_song = df.loc[match_idx]
                        dist_score = round(distances[0][i], 4)
                        
                        # Prevent the algorithm from recommending the exact same song
                        if matched_song['track_name'].lower() == source_info['track_name'].lower():
                            continue 
                            
                        matches_shown += 1
                        
                        # UI formatting to highlight the best match
                        if matches_shown == 1:
                            st.info(f"🥇 **#1 CLOSEST MATCH:** {matched_song['track_name']} by {matched_song['artists_name']} ({int(matched_song['year'])})  \n*Distance Score: {dist_score}*")
                        elif matches_shown == 2:
                            st.success(f"🥈 **#2 Match:** {matched_song['track_name']} by {matched_song['artists_name']} ({int(matched_song['year'])})  \n*Distance Score: {dist_score}*")
                        else:
                            st.warning(f"🥉 **#3 Match:** {matched_song['track_name']} by {matched_song['artists_name']} ({int(matched_song['year'])})  \n*Distance Score: {dist_score}*")
