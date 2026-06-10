# Imported the needed libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import itertools

# Loaded the master cleaned dataset
df = pd.read_csv("spotify_master_cleaned.csv")
features = ['acousticness', 'danceability', 'energy', 'loudness', 'tempo', 'valence']

# Scaled the audio features to a 0–1 range
scaler = MinMaxScaler()
df_scaled = df.copy()
df_scaled[features] = scaler.fit_transform(df[features])

# The Time Machine recommendation function
def get_recommendations(source_song, target_decade, artist_name=""):
    """
    Finds the top 3 sonic matches for a given song in a target decade.

    Parameters
    ----------
    source_song : str Track name to search for
    target_decade : int e.g. 1980 for the 1980s
    artist_name : str Optional artist name to narrow the search

    Returns
    -------
    dict with keys:
        source_info — details of the matched input track
        results — list of 3 recommended tracks with scores
        metrics — all four evaluation metrics
    """

    # Finding the source song
    mask = df_scaled['track_name'].str.contains(source_song, case=False, regex=False)

    if artist_name.strip():
        mask = mask & df_scaled['artists_name'].str.contains(artist_name, case=False, regex=False)

    match = df_scaled[mask]

    if match.empty:
        return {"error": f"'{source_song}' not found. Try a different title or leave the artist field blank."}

    # Getting the scaled feature vector for the source song
    source_vector = match[features].iloc[0].values.reshape(1, -1)
    source_info   = df[mask].iloc[0]

    # Filter to the target decade
    decade_pool = df_scaled[
        (df_scaled['year'] >= target_decade) &
        (df_scaled['year'] < target_decade + 10)]

    if decade_pool.empty:
        return {"error": f"No tracks found in the {target_decade}s."}

    # STEP 1: k-NN search using Euclidean distance
    # This finds the 3 closest tracks in the 6-dimensional feature space
    # N_neighbors=4 in case the searched song itself appears in the decade pool.

    knn = NearestNeighbors(n_neighbors=4, metric='euclidean')
    knn.fit(decade_pool[features])
    distances, indices = knn.kneighbors(source_vector)

    # Collect the top 3 recommended tracks and their feature vectors
    results = []
    rec_vectors = []

    for i in range(1, 4):
        match_idx = decade_pool.index[indices[0][i]]
        matched_song = df.loc[match_idx]
        dist_score = round(float(distances[0][i]), 4)

        results.append({
            'track_name': matched_song['track_name'],
            'artists_name': matched_song['artists_name'],
            'year': int(matched_song['year']),
            'euclidean_distance': dist_score,
            'features': {f: round(float(matched_song[f]), 4) for f in features}})

        rec_vectors.append(decade_pool[features].iloc[indices[0][i]].values)

    # STEP 2: Cosine Similarity (source vs each result)
    # Verifies that the proportional shape of each result's feature vector aligns with the source, not just that
    # the absolute values are close.

    cos_sims = [
        cosine_similarity(source_vector, vec.reshape(1, -1))[0][0]
        for vec in rec_vectors]
    avg_cosine = round(float(np.mean(cos_sims)), 4)

    # Add per-result cosine similarity to each result dict
    for i, score in enumerate(cos_sims):
        results[i]['cosine_similarity'] = round(float(score), 4)

    # STEP 3: Intra-List Similarity (ILS)
    # Measures pairwise cosine similarity between the 3 results.
    # A high ILS means the recommendations are sonically coherent as a set, not just individually close to input.

    pairs = list(itertools.combinations(rec_vectors,2))
    ils_scores = [
        cosine_similarity(p[0].reshape(1, -1), p[1].reshape(1, -1))[0][0]
        for p in pairs]
    ils = round(float(np.mean(ils_scores)), 4)

    # Random baseline (for evaluation comparison)
    # Simulates picking 3 random tracks from the same decade to show k-NN outperforms chance selection.
    random_sample = decade_pool[features].sample(n=3, random_state=42).values
    random_dists = [float(np.linalg.norm(source_vector - v)) for v in random_sample]
    random_baseline = round(float(np.mean(random_dists)), 4)
    avg_euclidean = round(float(np.mean(distances[0][1:4])), 4)
    pct_below_random = round((1 - avg_euclidean / random_baseline) * 100, 1)

    # Bundled all metrics
    metrics = {
        'avg_euclidean_distance': avg_euclidean,
        'random_baseline': random_baseline,
        'pct_below_random': pct_below_random,
        'avg_cosine_similarity': avg_cosine,
        'intra_list_similarity': ils,}

    return {
        'source_info': source_info,
        'results': results,
        'metrics': metrics,}


# Quick test
if __name__ == "__main__":
    output = get_recommendations("Blinding Lights", 1980)

    src = output['source_info']
    print(f"Source: {src['track_name']} by {src['artists_name']} ({int(src['year'])})")

    print(f"\nTop 3 matches from the 1980s:")
    for r in output['results']:
        print(f"* {r['track_name']} by {r['artists_name']} ({r['year']})")
        print(f"Euclidean Distance : {r['euclidean_distance']}")
        print(f"Cosine Similarity : {r['cosine_similarity']}")

    print(f"\nOverall Metrics:")
    for k, v in output['metrics'].items():
        print(f"{k}: {v}")
