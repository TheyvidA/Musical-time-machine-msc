# Imported the needed library
import pandas as pd

# Loaded both spotify datasets
df_historical = pd.read_csv("Spotify_1970-2020_Fixed.csv", encoding='latin-1')
df_modern = pd.read_csv("spotify_tracks.csv", encoding='latin-1')


# Filtered each dataset to the correct year range
#    DS1 → 1970 to 2019
#    DS2 → 2020 to 2024
df_1970_2019 = df_historical[(df_historical['year'] >= 1970) & (df_historical['year'] <= 2019)]
df_2020_2024 = df_modern[(df_modern['year'] >= 2020) & (df_modern['year'] <= 2024)]

print(f"Historical (1970-2019): {len(df_1970_2019):,} tracks")
print(f"Modern (2020-2024): {len(df_2020_2024):,} tracks")

# Keep only the columns we need
required_columns = [
    'track_name', 'artists_name', 'year',
    'acousticness', 'danceability', 'energy',
    'loudness', 'tempo', 'valence']

df_hist_ready = df_1970_2019[required_columns]
df_mod_ready = df_2020_2024[required_columns]

# Merged the two datasets
df_master = pd.concat([df_hist_ready, df_mod_ready], ignore_index=True)

# Removed duplicate tracks
df_master = df_master.drop_duplicates(subset=['track_name', 'artists_name', 'year'])
print(f"After de-duplication: {len(df_master):,} tracks")

# Removed rows with invalid or missing audio feature values
df_clean = df_master[
    (df_master['valence'] >= 0.0) &
    (df_master['energy'] >= 0.0) &
    (df_master['acousticness'] >= 0.0) &
    (df_master['loudness'] > -100.0)].dropna(subset=required_columns)

print(f"After cleaning: {len(df_clean):,} tracks")
print(f"Year range: {int(df_clean['year'].min())} - {int(df_clean['year'].max())}")

# Save the final master file
df_clean.to_csv('spotify_master_cleaned.csv', index=False)
print("Saved: spotify_master_cleaned.csv")
