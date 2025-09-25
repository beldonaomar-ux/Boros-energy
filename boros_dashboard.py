import streamlit as st
import pandas as pd

# Page setup
st.set_page_config(page_title="Boros Dashboard", layout="wide")
st.title("🔴 Boros Energy Winrate Dashboard")

# Load data
df = pd.read_csv("boros_energy_winrate_predictions.csv")

# Show overall winrate chart
st.subheader("⚡ Predicted Winrate Over Time")
if "Predicted Winrate" in df.columns:
    st.line_chart(df["Predicted Winrate"])
else:
    st.warning("Predicted Winrate column not found.")

# Detect matchup columns
matchup_cols = [col for col in df.columns if col.startswith("Winrate_vs_")]

# Matchup dropdown
if matchup_cols:
    selected = st.selectbox("🧩 Choose Opponent Archetype", matchup_cols)
    st.subheader(f"📊 Winrate vs {selected.replace('Winrate_vs_', '').replace('_', ' ')}")
    st.line_chart(df[selected])
else:
    st.warning("No matchup data found in the CSV.")

# View all matchup winrates
if matchup_cols:
    st.subheader("📋 All Matchup Winrates")
    avg_winrates = df[matchup_cols].mean().sort_values(ascending=False)
    st.bar_chart(avg_winrates)

    # Top and bottom matchups
    st.subheader("🔥 Top 5 Matchups")
    st.bar_chart(avg_winrates.head(5))

    st.subheader("💀 Bottom 5 Matchups")
    st.bar_chart(avg_winrates.tail(5))

    # Optional: show as table
    with st.expander("📄 View All Matchups as Table"):
        st.dataframe(avg_winrates)

# 🧠 Deck Trait Diagnostics
st.subheader("🧠 Deck Trait Diagnostics")

# Define archetype categories
interaction_heavy = ["Control", "Midrange", "Tempo"]
low_interaction = ["Aggro", "Combo", "Ramp"]

# Extract archetype names from column headers
archetypes = [col.replace("Winrate_vs_", "").replace("_", " ") for col in matchup_cols]

# Resilience: winrate vs. interaction-heavy decks
resilience_cols = [col for col in matchup_cols if any(tag.lower() in col.lower() for tag in interaction_heavy)]
resilience_score = df[resilience_cols].mean().mean() if resilience_cols else 0

# Explosiveness: winrate vs. low-interaction decks
explosive_cols = [col for col in matchup_cols if any(tag.lower() in col.lower() for tag in low_interaction)]
explosiveness_score = df[explosive_cols].mean().mean() if explosive_cols else 0

# Versatility: winrate spread across archetypes
versatility_score = df[matchup_cols].mean().std()

# Adaptability: winrate variance over time
if "Predicted Winrate" in df.columns:
    adaptability_score = df["Predicted Winrate"].std()
else:
    adaptability_score = 0

# Late-game strength: winrate vs. Control
control_cols = [col for col in matchup_cols if "control" in col.lower()]
late_game_score = df[control_cols].mean().mean() if control_cols else 0

# Display trait scores
trait_scores = {
    "Resilience": resilience_score,
    "Explosiveness": explosiveness_score,
    "Versatility": versatility_score,
    "Adaptability": adaptability_score,
    "Late Game": late_game_score
}
trait_df = pd.DataFrame(trait_scores, index=["Score"]).T
st.bar_chart(trait_df)

# Raw data viewer
with st.expander("📄 Full Dataset"):
    st.dataframe(df)