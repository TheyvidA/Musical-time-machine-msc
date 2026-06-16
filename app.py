# Imported the needed libraries
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time

# Import the recommendation engine from modelling.py
from model import get_recommendations, df, features, df_scaled

# Page configuration
st.set_page_config(page_title="The Musical Time Machine", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .metric-box {
        background: #f0f4fa;
        border-radius: 10px;
        padding: 14px 18px;
        text-align: center;}
    .metric-num { font-size: 1.6rem; font-weight: 700; color: #1a3c6e; }
    .metric-label { font-size: 0.75rem; color: #666; margin-top: 3px; }
    .metric-note { font-size: 0.68rem; color: #999; margin-top: 2px; }
    .result-card {
        background: #ffffff;
        border: 1px solid #dde3ee;
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 10px; }
    .result-rank { font-size: 0.72rem; font-weight: 600; color: #2E75B6; text-transform: uppercase; letter-spacing: .04em; }
    .result-title { font-size: 1rem; font-weight: 600; color: #1a3c6e; margin-top: 2px; }
    .result-sub { font-size: 0.83rem; color: #555; margin-top: 2px; }
    .result-scores { font-size: 0.78rem; color: #888; margin-top: 5px; }
    .source-card {
        background: #eef3fb;
        border: 1px solid #c0d0ea;
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 14px; }
    .disclaimer {
        font-size: 0.73rem; color: #aaa;
        border-top: 1px solid #eee;
        padding-top: 10px; margin-top: 10px;
        line-height: 1.6; }
</style>
""", unsafe_allow_html=True)

DECADE_OPTIONS = {
    "The 1970s": 1970, "The 1980s": 1980, "The 1990s": 1990,
    "The 2000s": 2000, "The 2020s": 2020}

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
    df['decade'] = (df['year'] // 10 * 10).astype(int)
    decade_counts = df['decade'].value_counts().sort_index()
    for dec, cnt in decade_counts.items():
        label = f"{int(dec)}s"
        st.progress(int(cnt / decade_counts.max() * 100), text=f"{label}: {cnt:,}")

    st.divider()
    st.caption("**Audio features used:**")
    st.caption("Acousticness · Danceability · Energy\nLoudness · Tempo · Valence")
    st.divider()
    st.caption("**Algorithm:** K-Nearest Neighbours (Euclidean distance)")
    st.caption("**Metrics:** Euclidean Distance · Cosine Similarity · ILS")

# Header
st.title("⌚ The Musical Time Machine ⌚")
st.write("Search for a song by name and artist, select a target decade, and find its closest sonic match from the past.")
st.divider()

# Search inputs
col1, col2, col3 = st.columns([2.5, 2.5, 1.5])
with col1:
    track_input = st.text_input("Song name", placeholder="e.g. Blinding Lights")
with col2:
    artist_input = st.text_input("Artist name (optional)", placeholder="e.g. The Weeknd")
with col3:
    target_label = st.selectbox("Target decade", list(DECADE_OPTIONS.keys()))

target_decade = DECADE_OPTIONS[target_label]
run = st.button("Travel Through Time")
st.divider()

# Run the Time Machine
if run:
    if not track_input.strip():
        st.warning("Please enter a song name to continue.")
    else:
        with st.spinner("Searching across the decades..."):
            t0 = time.perf_counter()
            output = get_recommendations(track_input, target_decade, artist_input)
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        # Error handling
        if "error" in output:
            st.error(output["error"])

        else:
            src = output['source_info']
            results = output['results']
            m = output['metrics']

            latency_colour = "green" if elapsed_ms < 3000 else "red"
            st.success(f"Time jump successful!  ·  Response time: :{latency_colour}[**{elapsed_ms} ms**]")

            # ROW 1 — Four evaluation metric cards
            # These come directly from get_recommendations() in modelling.py — Euclidean distance found the matches, cosine similarity and ILS validate them.
            c1, c2, c3, c4 = st.columns(4)

            c1.markdown(f"""
            <div class="metric-box">
                <div class="metric-num">{m['avg_euclidean_distance']}</div>
                <div class="metric-label">Avg Euclidean Distance</div>
                <div class="metric-note">lower = closer match</div>
            </div>""", unsafe_allow_html=True)

            c2.markdown(f"""
            <div class="metric-box">
                <div class="metric-num">{m['avg_cosine_similarity']}</div>
                <div class="metric-label">Avg Cosine Similarity</div>
                <div class="metric-note">closer to 1.0 = better</div>
            </div>""", unsafe_allow_html=True)

            c3.markdown(f"""
            <div class="metric-box">
                <div class="metric-num">{m['pct_below_random']}%</div>
                <div class="metric-label">Closer than Random</div>
                <div class="metric-note">vs random decade selection</div>
            </div>""", unsafe_allow_html=True)

            c4.markdown(f"""
            <div class="metric-box">
                <div class="metric-num">{m['intra_list_similarity']}</div>
                <div class="metric-label">Intra-List Similarity</div>
                <div class="metric-note">result set coherence</div>
            </div>""", unsafe_allow_html=True)

            st.write("")

            # ROW 2 — Source track | Recommendations
            left, right = st.columns(2)

            # Left: source track + radar chart
            with left:
                st.markdown("**Your Track**")
                st.markdown(f"""
                <div class="source-card">
                    <div class="result-title">{src['track_name']}</div>
                    <div class="result-sub">{src['artists_name']} · {int(src['year'])}</div>
                </div>""", unsafe_allow_html=True)

                # Radar chart — source track vs top match overlaid

                src_vals = [float(src[f]) for f in features]
                top_vals = [results[0]['features'][f] for f in features]

                radar = go.Figure()

                # Source track polygon
                radar.add_trace(go.Scatterpolar(
                    r=src_vals + [src_vals[0]],
                    theta=features + [features[0]],
                    fill='toself',
                    fillcolor='rgba(46,117,182,0.15)',
                    line=dict(color='#2E75B6', width=2.5),
                    name=f"{src['track_name'][:25]} ({int(src['year'])})"
                ))

                # Top match polygon overlaid
                radar.add_trace(go.Scatterpolar(
                    r=top_vals + [top_vals[0]],
                    theta=features + [features[0]],
                    fill='toself',
                    fillcolor='rgba(29,158,117,0.15)',
                    line=dict(color='#1D9E75', width=2.5),
                    name=f"{results[0]['track_name'][:25]} ({results[0]['year']})"
                ))

                radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2, font=dict(size=10)),
                    margin=dict(l=20, r=20, t=30, b=40),
                    height=320,
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(radar, use_container_width=True, key="radar")

            # Right: top 3 results + comparison bar chart
            with right:
                st.markdown(f"**Top 3 Matches from {target_label}**")

                for i, r in enumerate(results, 1):
                    st.markdown(f"""
                    <div class="result-card">
                        <div class="result-rank">Match {i}</div>
                        <div class="result-title">{r['track_name']}</div>
                        <div class="result-sub">{r['artists_name']} · {r['year']}</div>
                        <div class="result-scores">
                            Euclidean Distance: <b>{r['euclidean_distance']}</b>
                            &nbsp;·&nbsp;
                            Cosine Similarity: <b>{r['cosine_similarity']}</b>
                        </div>
                    </div>""", unsafe_allow_html=True)

                # Feature comparison bar chart: source vs top match
                top = results[0]
                bar = go.Figure()
                bar.add_trace(go.Bar(
                    name=f"{src['track_name'][:22]}… ({int(src['year'])})",
                    x=features,
                    y=[float(src[f]) for f in features],
                    marker_color='#2E75B6'
                ))
                bar.add_trace(go.Bar(
                    name=f"{top['track_name'][:22]}… ({top['year']})",
                    x=features,
                    y=[top['features'][f] for f in features],
                    marker_color='#1D9E75'
                ))
                bar.update_layout(
                    barmode='group',
                    title="Feature Comparison: Your Track vs Top Match",
                    yaxis=dict(range=[0, 1], title="Feature Value (0–1)"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=10)),
                    margin=dict(l=10, r=10, t=50, b=10),
                    height=290,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    yaxis_gridcolor='#eee'
                )
                st.plotly_chart(bar, use_container_width=True, key="bar")

            # Disclaimer
            st.markdown("""
            <div class="disclaimer">
            Recommendations are based on sonic similarity only — acousticness, danceability,
            energy, loudness, tempo, and valence. They carry no thematic, cultural, or historical
            equivalence. The dataset reflects a predominantly Western, English-language bias.
            </div>""", unsafe_allow_html=True)


# Sonic Evolution — EDA section
st.divider()
with st.expander("Sonic Evolution — How music has changed across decades (1970–2024)"):

    decade_agg    = df.groupby('decade')[features].mean().reset_index()
    decade_labels = [f"{int(d)}s" for d in decade_agg['decade']]

    feat_select = st.multiselect(
        "Select audio features to compare",
        options=features,
        default=['valence', 'energy', 'acousticness']
    )

    if feat_select:
        colours = ['#2E75B6','#E15759','#59A14F','#F28E2B','#B07AA1','#76B7B2']
        evo = go.Figure()
        for i, feat in enumerate(feat_select):
            evo.add_trace(go.Scatter(
                x=decade_labels, y=decade_agg[feat],
                mode='lines+markers', name=feat,
                line=dict(color=colours[i % len(colours)], width=2.5),
                marker=dict(size=8)
            ))
        evo.update_layout(
            title="Mean Audio Feature Value by Decade",
            yaxis=dict(range=[0, 1], title="Mean Value (0–1)", gridcolor='#eee'),
            xaxis_title="Decade",
            height=380,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(evo, use_container_width=True)

    # Loudness War
    st.markdown("**The Loudness War** — Average recorded loudness (dB) has risen steadily since the 1970s (Devine, 2013).")
    loud_agg = df.groupby('decade')['loudness'].mean().reset_index()
    loud = go.Figure(go.Bar(
        x=[f"{int(d)}s" for d in loud_agg['decade']],
        y=loud_agg['loudness'],
        marker_color='#2E75B6',
        text=[f"{v:.1f} dB" for v in loud_agg['loudness']],
        textposition='outside'
    ))
    loud.update_layout(
        yaxis_title="Mean Loudness (dB)",
        height=320,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis_gridcolor='#eee'
    )
    st.plotly_chart(loud, use_container_width=True)
