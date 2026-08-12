from model import get_recommendations

# This file runs a set of test queries through the Time Machine and prints a full evaluation report for each one.
# It imports get_recommendations() from model.py — so all three metrics (Euclidean distance, Cosine Similarity, ILS)
# come from the same function that powers the Streamlit app.
# There is no duplication. One function, one source of truth.

def run_evaluation(source_song, target_decade, artist_name=""):
    print(f" EVALUATING: '{source_song}' -> {target_decade}s --")

    output = get_recommendations(source_song, target_decade, artist_name)

    if "error" in output:
        print(output["error"], "\n")
        return

    src = output['source_info']
    print(f"Source track : {src['track_name']} by {src['artists_name']} ({int(src['year'])})")
    print()

    # Per-result scores
    print("Results:")
    for i, r in enumerate(output['results'], 1):
        print(f"  {i}. {r['track_name']} by {r['artists_name']} ({r['year']})")
        print(f"     Euclidean Distance : {r['euclidean_distance']}  (lower = closer match)")
        print(f"     Cosine Similarity  : {r['cosine_similarity']}   (closer to 1.0 = better alignment)")

    # Overall metrics
    m = output['metrics']
    print()
    print("Overall Evaluation Metrics:")
    print(f" 1. Avg Euclidean Distance : {m['avg_euclidean_distance']} | Random baseline: {m['random_baseline']}")
    print(f" 2. Improvement over random : {m['pct_below_random']}% closer than random selection")
    print(f" 3. Avg Cosine Similarity : {m['avg_cosine_similarity']} (source vs each result)")
    print(f" 4. Intra-List Similarity : {m['intra_list_similarity']} (coherence across the 3 results)")
    print()


# Run the evaluation test cases
run_evaluation("Blinding Lights", 1980)
run_evaluation("Levitating", 1970)
run_evaluation("Shape of You", 1990)
run_evaluation("Bohemian Rhapsody", 2010)
