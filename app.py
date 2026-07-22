import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
import os
import itertools
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

# Page configuration
st.set_page_config(page_title="The Musical Time Machine", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 1.8rem; padding-bottom: 2rem; }
    .metric-box {
        background: #f0f4fa; border-radius: 10px;
        padding: 14px 18px; text-align: center;
    }
    .metric-num { font-size: 1.55rem; font-weight: 700; color: #1a3c6e; }
    .metric-label { font-size: 0.74rem; color: #666; margin-top: 3px; }
    .metric-note { font-size: 0.67rem; color: #999; margin-top: 2px; }
    .result-card {
        background: #fff; border: 1px solid #dde3ee;
        border-radius: 10px; padding: 14px 16px; margin-bottom: 10px;
    }
    .result-rank { font-size: 0.7rem; font-weight: 600; color: #2E75B6;
                    text-transform: uppercase; letter-spacing:.04em; }
    .result-title { font-size: 1rem; font-weight: 600; color: #1a3c6e; margin-top:2px; }
    .result-sub { font-size: 0.82rem; color: #555; margin-top: 2px; }
    .result-scores{ font-size: 0.77rem; color: #888; margin-top: 5px; }
    .source-card {
        background: #eef3fb; border: 1px solid #c0d0ea;
        border-radius: 10px; padding: 14px 16px; margin-bottom: 14px;
    }
    .why-table th { background:#f0f4fa; font-size:0.78rem; padding:6px 10px; }
    .why-table td { font-size:0.78rem; padding:6px 10px; }
    .disclaimer {
        font-size:0.72rem; color:#aaa;
        border-top:1px solid #eee; padding-top:10px; margin-top:10px; line-height:1.6;
    }
    .section-connector {
        background: #f8faff; border-left: 3px solid #2E75B6;
        padding: 10px 14px; border-radius: 0 8px 8px 0;
        font-size: 0.83rem; color: #444; margin: 12px 0;
    }
</style>
""", unsafe_allow_html=True)

FEATURES = ['acousticness', 'danceability', 'energy', 'loudness', 'tempo', 'valence']
DECADE_OPTIONS = {
    "The 1970s": 1970, "The 1980s": 1980, "The 1990s": 1990,
    "The 2000s": 2000, "The 2010s": 2010, "The 2020s" : 2020}

# Load and cache data
@st.cache_data
def load_data():
    # Get the exact directory where app.py is located
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Build the path to the CSV file dynamically
    csv_path = os.path.join(current_dir, 'spotify_master_cleaned.csv')
    
    # Load the path
    df = pd.read_csv(csv_path)
    df['decade'] = (df['year'] // 10 * 10).astype(int)
    return df

df = load_data()
scaler = MinMaxScaler()
df_scaled = df.copy()
df_scaled[FEATURES] = scaler.fit_transform(df[FEATURES])

# Pre-compute decade averages once for the trend connector section
decade_agg = df.groupby('decade')[FEATURES].mean().reset_index()

# Sidebar
with st.sidebar:
    st.markdown("### The Musical Time Machine")
    st.divider()
    st.markdown("**Dataset**")
    st.metric("Total Tracks", f"{len(df):,}")
    st.metric("Year Coverage", "1970 – 2024")
    st.metric("Decades", "6")
    st.divider()
    st.markdown("**Tracks per Decade**")
    decade_counts = df['decade'].value_counts().sort_index()
    for dec, cnt in decade_counts.items():
        st.progress(int(cnt / decade_counts.max() * 100),
                    text=f"{int(dec)}s: {cnt:,}")
    st.divider()
    st.caption("**Audio features:** acousticness · danceability · energy · loudness · tempo · valence")
    st.caption("**Algorithm:** k-Nearest Neighbours (Euclidean distance)")
    st.caption("**Validation:** Cosine Similarity · ILS · Random Baseline")

# Header + inputs
st.title("🎵The Musical Time Machine")
st.write("Search for a song by name and artist, select a target decade, and discover its closest sonic match from the past.")
st.divider()

col1, col2, col3 = st.columns([2.5, 2.5, 1.5])
with col1:
    track_input = st.text_input("Song name", placeholder="e.g. Blinding Lights")
with col2:
    artist_input = st.text_input("Artist name (optional)", placeholder="e.g. The Weeknd")
with col3:
    target_label = st.selectbox("Target decade", list(DECADE_OPTIONS.keys()))

target_decade = DECADE_OPTIONS[target_label]
run = st.button("🚀 Travel Through Time")
st.divider()

# Time Machine engine
if run:
    if not track_input.strip():
        st.warning("Please enter a song name to continue.")
    else:
        with st.spinner("Searching across the decades..."):
            t0 = time.perf_counter()

            # Find source song
            mask = df_scaled['track_name'].str.contains(
                track_input, case=False, regex=False)
            if artist_input.strip():
                mask = mask & df_scaled['artists_name'].str.contains(
                    artist_input, case=False, regex=False)

            match = df_scaled[mask]

            if match.empty:
                st.error(
                    f"No match found for **'{track_input}'**"
                    + (f" by **'{artist_input}'**" if artist_input.strip() else "")
                    + ". Try checking the spelling or leave the artist field blank.")
                st.stop()

            source_vector = match[FEATURES].iloc[0].values.reshape(1, -1)
            source_row = df[mask].iloc[0]

            # Filter to target decade
            decade_mask = ((df_scaled['year'] >= target_decade) &
                           (df_scaled['year'] <  target_decade + 10))
            decade_pool = df_scaled[decade_mask]

            # k-NN using Euclidean distance (operational search metric)
            knn = NearestNeighbors(n_neighbors=4, metric='euclidean')
            knn.fit(decade_pool[FEATURES])
            distances, indices = knn.kneighbors(source_vector)

            elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

            # Collect top 3 results + their feature vectors
            results = []
            rec_vectors = []
            for i in range(1, 4):
                idx = decade_pool.index[indices[0][i]]
                row = df.loc[idx]
                dist = round(float(distances[0][i]), 4)
                results.append({
                    'track_name': row['track_name'],
                    'artists_name': row['artists_name'],
                    'year': int(row['year']),
                    'euclidean_distance': dist,
                    'features': {f: round(float(row[f]), 4) for f in FEATURES}
                })
                rec_vectors.append(decade_pool[FEATURES].iloc[indices[0][i]].values)

            # Compute validation metrics
            cos_sims = [
                cosine_similarity(source_vector, v.reshape(1,-1))[0][0]
                for v in rec_vectors
            ]
            avg_cosine = round(float(np.mean(cos_sims)), 4)
            for i, s in enumerate(cos_sims):
                results[i]['cosine_similarity'] = round(float(s), 4)

            pairs = list(itertools.combinations(rec_vectors, 2))
            ils_scores = [cosine_similarity(p[0].reshape(1,-1),
                          p[1].reshape(1,-1))[0][0] for p in pairs]
            ils = round(float(np.mean(ils_scores)), 4)

            random_sample = decade_pool[FEATURES].sample(n=3, random_state=42).values
            random_dists = [float(np.linalg.norm(source_vector - v))
                               for v in random_sample]
            random_baseline = round(float(np.mean(random_dists)), 4)
            avg_euclidean = round(float(np.mean(distances[0][1:4])), 4)
            pct_below = round((1 - avg_euclidean / random_baseline) * 100, 1)

        # SECTION A — Status + metric cards
        lc = "green" if elapsed_ms < 3000 else "red"
        st.success(
            f"Time jump successful! ✅  ·  Response time: :{lc}[**{elapsed_ms} ms**]")

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"""<div class="metric-box">
            <div class="metric-num">{avg_euclidean}</div>
            <div class="metric-label">Avg Euclidean Distance</div>
            <div class="metric-note">lower = closer match</div>
        </div>""", unsafe_allow_html=True)
        c2.markdown(f"""<div class="metric-box">
            <div class="metric-num">{avg_cosine}</div>
            <div class="metric-label">Avg Cosine Similarity</div>
            <div class="metric-note">closer to 1.0 = better</div>
        </div>""", unsafe_allow_html=True)
        c3.markdown(f"""<div class="metric-box">
            <div class="metric-num">{pct_below}%</div>
            <div class="metric-label">Closer than Random</div>
            <div class="metric-note">vs random decade pick</div>
        </div>""", unsafe_allow_html=True)
        c4.markdown(f"""<div class="metric-box">
            <div class="metric-num">{ils}</div>
            <div class="metric-label">Intra-List Similarity</div>
            <div class="metric-note">result set coherence</div>
        </div>""", unsafe_allow_html=True)

        st.write("")

        # SECTION B — Source track (left) | Top 3 matches (right)
        left, right = st.columns(2)

        with left:
            st.markdown("**Your Track**")
            st.markdown(f"""<div class="source-card">
                <div class="result-title">{source_row['track_name']}</div>
                <div class="result-sub">{source_row['artists_name']} · {int(source_row['year'])}</div>
            </div>""", unsafe_allow_html=True)

            # Radar: source vs top match overlaid
            src_vals = [float(source_row[f]) for f in FEATURES]
            top_vals = [results[0]['features'][f] for f in FEATURES]

            radar = go.Figure()
            radar.add_trace(go.Scatterpolar(
                r=src_vals + [src_vals[0]],
                theta=FEATURES + [FEATURES[0]],
                fill='toself',
                fillcolor='rgba(46,117,182,0.15)',
                line=dict(color='#2E75B6', width=2.5),
                name=f"{source_row['track_name'][:22]} ({int(source_row['year'])})"
            ))
            radar.add_trace(go.Scatterpolar(
                r=top_vals + [top_vals[0]],
                theta=FEATURES + [FEATURES[0]],
                fill='toself',
                fillcolor='rgba(29,158,117,0.15)',
                line=dict(color='#1D9E75', width=2.5),
                name=f"{results[0]['track_name'][:22]} ({results[0]['year']})"
            ))
            radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.25,
                            font=dict(size=10)),
                margin=dict(l=20, r=20, t=30, b=50),
                height=320,
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(radar, use_container_width=True, key="radar")

        with right:
            st.markdown(f"**Top 3 Matches from {target_label}**")
            for i, r in enumerate(results, 1):
                st.markdown(f"""<div class="result-card">
                    <div class="result-rank">Match {i}</div>
                    <div class="result-title">{r['track_name']}</div>
                    <div class="result-sub">{r['artists_name']} · {r['year']}</div>
                    <div class="result-scores">
                        Euclidean Distance: <b>{r['euclidean_distance']}</b>
                        &nbsp;·&nbsp;
                        Cosine Similarity: <b>{r['cosine_similarity']}</b>
                    </div>
                </div>""", unsafe_allow_html=True)

        st.divider()

        # SECTION C — Why this match? (feature breakdown table)
        # Shows exactly why the top result was chosen, making
        # the recommendation system transparent and verifiable.
        st.markdown("#### Why This Match?")
        st.caption(
            f"Feature-by-feature comparison between **{source_row['track_name']}** "
            f"and the top match **{results[0]['track_name']}** — "
            f"showing how close each audio dimension is.")

        top = results[0]
        why_rows = []
        for f in FEATURES:
            src_val = round(float(source_row[f]), 3)
            top_val = round(top['features'][f], 3)
            diff = abs(src_val - top_val)
            # Similarity as a percentage (0 diff = 100%, diff of 1 = 0%)
            sim_pct = round((1 - diff) * 100, 1)
            bar = "🟩" * int(sim_pct // 20) + "⬜" * (5 - int(sim_pct // 20))
            why_rows.append({
                "Feature": f.capitalize(),
                "Your Track": src_val,
                "Top Match": top_val,
                "Difference": round(diff, 3),
                "Similarity": f"{bar}  {sim_pct}%"
            })

        why_df = pd.DataFrame(why_rows)
        st.dataframe(
            why_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Feature":    st.column_config.TextColumn("Feature"),
                "Your Track": st.column_config.NumberColumn("Your Track", format="%.3f"),
                "Top Match":  st.column_config.NumberColumn("Top Match",  format="%.3f"),
                "Difference": st.column_config.NumberColumn("Difference", format="%.3f"),
                "Similarity": st.column_config.TextColumn("Similarity"),
            }
        )

        st.divider()

        # SECTION D — Bigger trends connector
        # Shows where the searched song sits in the decade-level
        # trend lines, connecting the individual result to the
        # broader sonic evolution story in the EDA.
        st.markdown("#### Where Does Your Track Fit in the Bigger Picture?")
        st.markdown(
            f'<div class="section-connector">'
            f'The chart below shows how each audio feature has shifted across decades. '
            f'The <b style="color:#E63946">red dot</b> marks where '
            f'<b>{source_row["track_name"]}</b> ({int(source_row["year"])}) sits '
            f'on each trend line — showing whether it is ahead of, behind, or in line '
            f'with its era.</div>',
            unsafe_allow_html=True
        )

        feat_select = st.multiselect(
            "Select features to explore",
            options=FEATURES,
            default=['energy', 'valence', 'acousticness'],
            key="trend_select"
        )

        if feat_select:
            decade_labels  = [f"{int(d)}s" for d in decade_agg['decade']]
            src_decade_label = f"{int((source_row['year'] // 10) * 10)}s"

            # High-contrast, colourblind-friendly palette — each feature
            # gets a distinct colour, line style AND marker shape so they
            # are distinguishable even in greyscale print
            FEAT_STYLES = {
                'energy': dict(color='#E63946', dash='solid', symbol='circle',        width=3),
                'valence': dict(color='#2196F3', dash='solid', symbol='square',        width=3),
                'acousticness': dict(color='#4CAF50', dash='dash', symbol='diamond',       width=3),
                'danceability': dict(color='#FF9800', dash='dot', symbol='triangle-up',   width=3),
                'tempo': dict(color='#9C27B0', dash='dashdot',symbol='cross',         width=3),
                'loudness': dict(color='#00BCD4', dash='longdash',symbol='star',         width=3),
            }

            trend_fig = go.Figure()

            for feat in feat_select:
                s = FEAT_STYLES[feat]
                src_val = float(source_row[feat])

                # Decade trend line — thick, distinct style per feature
                trend_fig.add_trace(go.Scatter(
                    x=decade_labels,
                    y=decade_agg[feat],
                    mode='lines+markers',
                    name=feat.capitalize(),
                    line=dict(color=s['color'], width=s['width'], dash=s['dash']),
                    marker=dict(size=9, symbol=s['symbol'],
                                color=s['color'],
                                line=dict(color='white', width=1.5)),
                    hovertemplate=f"<b>{feat.capitalize()}</b><br>Decade: %{{x}}<br>Value: %{{y:.2f}}<extra></extra>"
                ))

                # Highlighted dot for the searched song
                # Larger, white-outlined, same colour as its feature line
                trend_fig.add_trace(go.Scatter(
                    x=[src_decade_label],
                    y=[src_val],
                    mode='markers+text',
                    marker=dict(
                        color=s['color'], size=20, symbol='star',
                        line=dict(color='white', width=2.5)
                    ),
                    text=[f" {source_row['track_name'][:20]}" if feat == feat_select[0] else ""],
                    textposition='middle right',
                    textfont=dict(size=11, color=s['color']),
                    showlegend=False,
                    hovertemplate=(
                        f"<b>⭐ Your Track</b><br>"
                        f"{source_row['track_name']}<br>"
                        f"{feat.capitalize()}: {src_val:.2f}<extra></extra>"
                    )
                ))

            # Vertical annotation band highlighting the searched decade
            src_idx = decade_labels.index(src_decade_label) if src_decade_label in decade_labels else None
            shapes = []
            annotations = []
            if src_idx is not None:
                shapes.append(dict(
                    type='rect',
                    xref='x', yref='paper',
                    x0=src_decade_label, x1=src_decade_label,
                    y0=0, y1=1,
                    line=dict(color='rgba(80,80,80,0.5)', width=2, dash='dot'),
                    fillcolor='rgba(255,235,59,0.08)'
                ))
                annotations.append(dict(
                    x=src_decade_label, y=1.06,
                    xref='x', yref='paper',
                    text=f"📍 {source_row['track_name'][:22]}'s decade",
                    showarrow=False,
                    font=dict(size=11, color='#555'),
                    bgcolor='rgba(255,235,59,0.6)',
                    borderpad=4
                ))

            trend_fig.update_layout(
                title=dict(
                    text=f"How Music Has Changed Since 1970  —  ⭐ marks where your track sits",
                    font=dict(size=14)
                ),
                yaxis=dict(
                    range=[0, 1],
                    title="Feature Value  (0 = low  →  1 = high)",
                    gridcolor='#e8e8e8',
                    tickformat='.1f'
                ),
                xaxis_title="Decade",
                height=440,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                shapes=shapes,
                annotations=annotations,
                legend=dict(
                    orientation="h",
                    yanchor="bottom", y=-0.28,
                    font=dict(size=12),
                    bgcolor='rgba(255,255,255,0.7)',
                    bordercolor='#ddd', borderwidth=1
                ),
                hovermode='x unified'
            )
            st.plotly_chart(trend_fig, use_container_width=True, key="trend_chart")

        # Loudness War chart
        # Problem with raw dB: values are negative (e.g. -12 dB) which
        # makes bars point downward and confuses non-technical readers.
        # This preserves the trend direction while reading intuitively
        # (bigger bar = louder, zero = baseline, positive = louder).
        st.markdown(
            "**The Loudness War** — Music has been mastered progressively louder "
            "every decade since the 1970s. The chart shows how much louder each "
            "decade is compared to the 1970s baseline."
        )

        src_loudness = float(source_row['loudness'])
        loud_agg = df.groupby('decade')['loudness'].mean().reset_index()
        baseline = loud_agg['loudness'].min() # quietest decade = 0
        loud_agg['loudness_relative'] = loud_agg['loudness'] - baseline
        src_loud_relative = src_loudness - baseline

        src_decade_label = f"{int((source_row['year'] // 10) * 10)}s"
        decade_x = [f"{int(d)}s" for d in loud_agg['decade']]

        # Colour bars on a gradient: light blue → dark blue = quieter → louder
        bar_colours = ['#90CAF9','#64B5F6','#42A5F5','#2196F3','#1565C0','#0D47A1']

        loud_fig = go.Figure()

        # Bars — relative loudness, coloured gradient quieter→louder
        loud_fig.add_trace(go.Bar(
            x=decade_x,
            y=loud_agg['loudness_relative'],
            marker=dict(
                color=bar_colours[:len(decade_x)],
                line=dict(color='white', width=1.5)
            ),
            text=[f"+{v:.1f} dB" for v in loud_agg['loudness_relative']],
            textposition='outside',
            textfont=dict(size=12, color='#333'),
            name='Decade average',
            hovertemplate="<b>%{x}</b><br>Louder than 1970s baseline: +%{y:.1f} dB<extra></extra>"
        ))

        # Highlight the searched song's decade bar in orange
        src_idx = decade_x.index(src_decade_label) if src_decade_label in decade_x else None
        if src_idx is not None:
            highlight_colours = list(bar_colours[:len(decade_x)])
            highlight_colours[src_idx] = '#FF6F00'
            loud_fig.update_traces(
                marker=dict(color=highlight_colours, line=dict(color='white', width=1.5)),
                selector=dict(type='bar')
            )

        # Horizontal dashed line — where the searched song sits
        loud_fig.add_hline(
            y=src_loud_relative,
            line=dict(color='#E63946', width=2.5, dash='dash'),
            annotation_text=f" ⭐ {source_row['track_name'][:25]} ({int(source_row['year'])}) — {src_loudness:.1f} dB",
            annotation_position="top left",
            annotation_font=dict(size=11, color='#E63946'),
            annotation_bgcolor='rgba(255,235,235,0.85)',
        )

        loud_fig.update_layout(
            yaxis=dict(
                title="Loudness increase compared to 1970s  (dB)",
                gridcolor='#e8e8e8',
                tickprefix='+',
                tickformat='.0f'
            ),
            xaxis_title="Decade",
            height=380,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            margin=dict(t=40, b=60),
            annotations=[dict(
                x=src_decade_label, y=src_loud_relative,
                xref='x', yref='y',
                text=f"Your track's decade",
                showarrow=True,
                arrowhead=2, arrowcolor='#FF6F00',
                arrowwidth=2,
                ax=0, ay=-45,
                font=dict(size=11, color='#FF6F00'),
                bgcolor='rgba(255,243,224,0.9)',
                bordercolor='#FF6F00', borderpad=4
            )] if src_idx is not None else []
        )
        st.plotly_chart(loud_fig, use_container_width=True, key="loudness_chart")

        st.markdown(
            '<div class="disclaimer">'
            'Recommendations are based on sonic similarity only — acousticness, danceability, '
            'energy, loudness, tempo, and valence. They carry no thematic, cultural, or historical '
            'equivalence. The dataset reflects a predominantly Western, English-language bias.'
            '</div>',
            unsafe_allow_html=True
        )
