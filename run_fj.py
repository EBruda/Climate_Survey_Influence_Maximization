import os
import sys
import json
import pickle
import time
from urllib.request import urlopen

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import plotly.express as px
from tqdm import tqdm
import reverse_geocoder as rg

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils"))
from utils import town_to_historical_county, town_to_planning_region

# =====================================================================
# CONFIGURATION — change YEAR to run for any year 2010-2024
# =====================================================================

YEAR = "2020"
TARGET_TOPIC = "worried"
EXCEL_PATH = "YCOM_2024_publicdata.xlsx"
GOWALLA_CHECKINS = "gowalla_data/Gowalla_totalCheckins.txt"
GOWALLA_CSV = "gowalla_data/gowalla_total.csv"
GOWALLA_EDGES = "gowalla_data/Gowalla_edges.txt"
FILTERED_EDGE_LIST = "gowalla_data/filtered_edge_list.txt"
NETWORK_GRAPH_DIR = "network_graphs"
RESULTS_DIR = "fj_model_results"
VIZ_DIR = "fixed_fj_visualizations"

# =====================================================================
# PART 1 — BUILD NETWORK GRAPH
# =====================================================================

def load_yale_data(year):
    target_oppose = TARGET_TOPIC + "Oppose"
    df = pd.read_excel(EXCEL_PATH, sheet_name="YCOM_2010_2024")
    df = df[df["GeoType"] == "county"].copy()
    df = df[(df["worried"] == TARGET_TOPIC) | (df["worried"] == target_oppose)].copy()
    df = df[df[f"x{year}"].notna()].copy()
    df = df[["GeoID", "GeoName", "GeoType", f"x{year}", "worried"]].copy()

    def split_name(row):
        parts = row["GeoName"].split(",")
        return pd.Series([parts[0].strip(), parts[1].strip()])

    df[["county", "state"]] = df.apply(split_name, axis=1)
    return df


def load_gowalla_checkins():
    if os.path.exists(GOWALLA_CSV):
        return pd.read_csv(GOWALLA_CSV)

    print("Reverse-geocoding Gowalla checkins (one-time, ~1-2 min)...")
    raw = pd.read_csv(GOWALLA_CHECKINS, sep="\t", header=None,
                      names=["userid", "timestamp", "latitude", "longitude", "spotid"])
    coords = list(zip(raw["latitude"], raw["longitude"]))
    results = []
    for i in tqdm(range(0, len(coords), 100000), desc="Geocoding batches"):
        results.extend(rg.search(coords[i:i + 100000], mode=2))

    raw["country"] = [r["cc"] for r in results]
    raw["state"] = [r["admin1"] for r in results]
    raw["county"] = [r["admin2"] for r in results]
    raw["city_name"] = [r["name"] for r in results]
    raw.to_csv(GOWALLA_CSV, index=False)
    return raw


def preprocess_gowalla(gowalla_total, year):
    gowalla_us = gowalla_total[gowalla_total["country"] == "US"].copy()

    # Connecticut: historical counties for years ≤ 2022, planning regions for 2023+
    ct_mask = gowalla_us["state"] == "Connecticut"
    if int(year) <= 2022:
        gowalla_us.loc[ct_mask, "county"] = (
            gowalla_us.loc[ct_mask, "city_name"].map(town_to_historical_county)
        )
    else:
        gowalla_us.loc[ct_mask, "county"] = (
            gowalla_us.loc[ct_mask, "city_name"].map(town_to_planning_region)
        )

    # Washington D.C.
    dc = gowalla_us["city_name"] == "Washington, D.C."
    gowalla_us.loc[dc, "county"] = "District of Columbia"
    gowalla_us.loc[dc, "state"] = "District of Columbia"

    # New York City — reverse geocoder sometimes returns blank county
    nyc = (gowalla_us["county"].fillna("") == "") & (gowalla_us["city_name"] == "New York City")
    gowalla_us.loc[nyc, "county"] = "New York County"

    # "City of X" → "X City"
    city_of = gowalla_us["county"].str.startswith("City of", na=False)
    gowalla_us.loc[city_of, "county"] = (
        gowalla_us.loc[city_of, "county"]
        .apply(lambda s: s.split("City of")[1].strip() + " City")
    )

    # Saint → St.
    saint = gowalla_us["county"].str.startswith("Saint", na=False)
    gowalla_us.loc[saint, "county"] = (
        gowalla_us.loc[saint, "county"].str.replace("Saint ", "St. ", n=1)
    )

    # Remaining edge cases
    gowalla_us.loc[gowalla_us["county"] == "Dona Ana County", "county"] = "Doña Ana County"
    gowalla_us.loc[gowalla_us["county"] == "Bronx", "county"] = "Bronx County"
    gowalla_us.loc[gowalla_us["county"] == "De Soto County", "county"] = "DeSoto County"
    bedford = (gowalla_us["county"] == "Bedford City") & (gowalla_us["state"] == "Virginia")
    gowalla_us.loc[bedford, "county"] = "Bedford County"

    assert len(gowalla_us[(gowalla_us["county"].fillna("") == "")]) == 0
    assert len(gowalla_us[(gowalla_us["state"].fillna("") == "")]) == 0

    return gowalla_us


def get_user_home_counties(gowalla_us):
    counts = gowalla_us.groupby(["userid", "county", "state"]).size().reset_index(name="Count")
    max_counts = counts.groupby("userid")["Count"].transform("max")
    return counts[counts["Count"] == max_counts]


def build_filtered_edge_list(us_users):
    if os.path.exists(FILTERED_EDGE_LIST):
        return
    filtered = []
    with open(GOWALLA_EDGES) as f:
        for line in f:
            u, v = map(int, line.strip().split())
            if u in us_users and v in us_users:
                filtered.append((u, v))
    with open(FILTERED_EDGE_LIST, "w") as f:
        for u, v in filtered:
            f.write(f"{u} {v}\n")


def build_network_graph(year):
    graph_path = os.path.join(NETWORK_GRAPH_DIR, f"network_graph_fj_{year}.pickle")
    if os.path.exists(graph_path):
        print(f"Loading cached graph for {year}...")
        with open(graph_path, "rb") as f:
            return pickle.load(f)

    os.makedirs(NETWORK_GRAPH_DIR, exist_ok=True)
    target_oppose = TARGET_TOPIC + "Oppose"

    print(f"Building network graph for {year}...")
    yale_county = load_yale_data(year)
    gowalla_total = load_gowalla_checkins()
    gowalla_us = preprocess_gowalla(gowalla_total, year)

    us_users = set(gowalla_us["userid"])
    build_filtered_edge_list(us_users)
    filtered_gowalla = get_user_home_counties(gowalla_us)

    G = nx.read_edgelist(FILTERED_EDGE_LIST, create_using=nx.Graph(), nodetype=int)

    skipped = 0
    for node in tqdm(G.nodes(), desc="Assigning opinions"):
        matches = filtered_gowalla[filtered_gowalla["userid"] == node]
        if matches.empty:
            continue

        user_county = matches["county"].iloc[0]
        user_state = matches["state"].iloc[0]
        G.nodes[node]["county"] = user_county
        G.nodes[node]["state"] = user_state

        yale_match = yale_county[
            (yale_county["state"] == user_state) &
            (yale_county["county"] == user_county)
        ]
        if yale_match.empty:
            print(f"  Warning: no Yale match for {user_county}, {user_state} — skipping")
            skipped += 1
            continue

        favor_row = yale_match[yale_match["worried"] == TARGET_TOPIC]
        oppose_row = yale_match[yale_match["worried"] == target_oppose]
        if favor_row.empty or oppose_row.empty:
            skipped += 1
            continue

        favor_pct = favor_row[f"x{year}"].iloc[0]
        oppose_pct = oppose_row[f"x{year}"].iloc[0]
        # Net opinion in [-1, 1]: positive = net support, negative = net opposition
        net_opinion = (favor_pct - oppose_pct) / 100.0

        G.nodes[node]["baseline_opinion"] = net_opinion
        G.nodes[node]["current_opinion"] = net_opinion
        G.nodes[node]["susceptibility"] = 0.5
        G.nodes[node]["GeoName"] = f"{user_county}, {user_state}"
        G.nodes[node]["GeoID"] = favor_row["GeoID"].iloc[0]

    if skipped:
        print(f"  Skipped {skipped} nodes with no matching Yale county.")

    for u, v in G.edges():
        same = (G.nodes[u].get("county") == G.nodes[v].get("county")
                and G.nodes[u].get("county") is not None)
        G[u][v]["weight"] = 1.0 if same else 0.5

    with open(graph_path, "wb") as f:
        pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)
    print(f"Saved graph to {graph_path}")
    return G


# =====================================================================
# PART 2 — FJ MODEL
# =====================================================================

def load_graph_and_matrices(G):
    nodes = list(G.nodes())
    print(f"Graph has {len(nodes)} nodes.")

    s = np.array([G.nodes[n].get("baseline_opinion", 0.0) for n in nodes])
    alpha = np.array([G.nodes[n].get("susceptibility", 0.5) for n in nodes])

    print("Building sparse weight matrix...")
    W_raw = nx.to_scipy_sparse_array(G, nodelist=nodes, weight="weight", format="csr")
    row_sums = np.array(W_raw.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    W = sp.diags(1.0 / row_sums) @ W_raw

    return nodes, W, s, alpha


def run_fj_simulation(W, s, alpha, max_iters=100, tolerance=1e-5):
    x = np.copy(s)
    for _ in range(max_iters):
        x_new = alpha * (W @ x) + (1.0 - alpha) * s
        if np.linalg.norm(x_new - x, ord=1) < tolerance:
            break
        x = x_new
    return x


def compute_influence_scores(W, alpha, steps=20):
    n = W.shape[0]
    v = np.ones(n)
    scores = np.zeros(n)
    for _ in range(steps):
        v = alpha * (W @ v)
        scores += v
    return scores


def get_top_candidates(scores, top_k=1000):
    return np.argsort(scores)[::-1][:top_k]


def greedy_influence_max(W, s, alpha, candidate_indices, k=10, max_iters=50):
    print("Running greedy influence maximization...")
    base_x = run_fj_simulation(W, s, alpha)
    base_score = np.mean(base_x)
    selected = []

    for step in range(k):
        best_gain = -np.inf
        best_node = None
        print(f"  Step {step + 1}/{k}")

        for idx in candidate_indices:
            if idx in selected:
                continue
            temp_s = np.copy(s)
            temp_alpha = np.copy(alpha)
            # Set intervention: pin node to maximum opinion (1.0 on the [-1,1] scale)
            temp_s[idx] = 1.0
            temp_alpha[idx] = 0.0
            score = np.mean(run_fj_simulation(W, temp_s, temp_alpha, max_iters=max_iters))
            gain = score - base_score
            if gain > best_gain:
                best_gain = gain
                best_node = idx

        selected.append(best_node)
        base_score += best_gain
        print(f"  Selected node {best_node} | Gain: {best_gain:.6f}")

    return selected


def evaluate_seed_set(seed_indices, W, s, alpha):
    temp_s = np.copy(s)
    temp_alpha = np.copy(alpha)
    for idx in seed_indices:
        temp_s[idx] = 1.0
        temp_alpha[idx] = 0.0
    final_x = run_fj_simulation(W, temp_s, temp_alpha)
    return np.mean(final_x), final_x


# =====================================================================
# PART 3 — VISUALIZATION
# =====================================================================

def plot_distribution_shift(x_baseline, x_final, year):
    os.makedirs(VIZ_DIR, exist_ok=True)
    plt.figure()
    plt.hist(x_final, bins=50, alpha=0.5, label="After Seeding")
    plt.hist(x_baseline, bins=50, alpha=0.5, label="Baseline")
    plt.xlabel("Net Opinion (favor − oppose) / 100")
    plt.ylabel("Number of Users")
    plt.title(f"Opinion Distribution Shift — {year}")
    plt.legend()
    plt.savefig(os.path.join(VIZ_DIR, f"{year}_opinion_distribution_shift.png"))
    plt.close()


def plot_choropleth(G, nodes, x_baseline, x_final, year):
    os.makedirs(VIZ_DIR, exist_ok=True)

    # FIX: opinions are already on the [-1, 1] net-opinion scale, so delta is
    # directly interpretable — do NOT divide by 100 again (that was the bug
    # causing all plots to look identical, compressing deltas to ~0.0002).
    delta_x = x_final - x_baseline

    data = [
        {
            "County": G.nodes[node].get("county", ""),
            "State": G.nodes[node].get("state", ""),
            "Opinion_Delta": delta_x[i],
        }
        for i, node in enumerate(nodes)
    ]
    df_raw = pd.DataFrame(data)
    df_county = df_raw.groupby(["State", "County"])["Opinion_Delta"].mean().reset_index()

    # FIPS lookup — filter to one row per county to avoid duplicates from the
    # worried/worriedOppose row structure in YCOM_2010_2024
    yale_lookup = pd.read_excel(EXCEL_PATH, sheet_name="YCOM_2010_2024")
    yale_lookup = yale_lookup[
        (yale_lookup["GeoType"] == "county") &
        (yale_lookup["worried"] == TARGET_TOPIC)
    ].copy()
    yale_lookup[["County", "State"]] = yale_lookup["GeoName"].str.split(",", expand=True)
    yale_lookup["County"] = yale_lookup["County"].str.strip()
    yale_lookup["State"] = yale_lookup["State"].str.strip()

    df_county = df_county.merge(
        yale_lookup[["County", "State", "GeoID"]], on=["County", "State"], how="left"
    )
    df_county = df_county.dropna(subset=["GeoID"])
    df_county["FIPS"] = df_county["GeoID"].astype(int).astype(str).str.zfill(5)
    df_county = df_county.drop(columns=["GeoID"])

    with urlopen(
        "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
    ) as response:
        counties_geojson = json.load(response)

    fig = px.choropleth(
        df_county,
        geojson=counties_geojson,
        locations="FIPS",
        color="Opinion_Delta",
        color_continuous_scale="Blues",
        range_color=[0, 0.5],
        scope="usa",
        hover_name="County",
        hover_data=["State", "Opinion_Delta"],
        title=f"Change in Opinion after Seeding — {year}",
    )
    fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
    # just save the html instead
    fig.write_html(os.path.join(VIZ_DIR, f"{year}_choropleth.html"))
    # fig.write_image(os.path.join(VIZ_DIR, f"change_opinion_{year}.png"))


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", default=YEAR, help="Yale data year (2010-2024)")
    year = parser.parse_args().year

    G = build_network_graph(year)
    nodes, W, s, alpha = load_graph_and_matrices(G)

    print(f"\n--- BASELINE SIMULATION ({year}) ---")
    t0 = time.time()
    x_baseline = run_fj_simulation(W, s, alpha)
    print(f"Baseline mean opinion: {np.mean(x_baseline):.4f}  ({time.time() - t0:.2f}s)")

    print("\n--- INFLUENCE SCORES ---")
    t0 = time.time()
    scores = compute_influence_scores(W, alpha, steps=20)
    candidate_indices = get_top_candidates(scores, top_k=1000)
    print(f"({time.time() - t0:.2f}s)")

    print("\n--- GREEDY INFLUENCE MAX ---")
    t0 = time.time()
    seeds = greedy_influence_max(W, s, alpha, candidate_indices, k=10)
    print(f"({time.time() - t0:.2f}s)")

    print("\n--- SEED NODE DETAILS ---")
    for seed_idx in seeds:
        node = nodes[seed_idx]
        print(
            f"  Node {seed_idx}: {G.nodes[node].get('county')}, "
            f"{G.nodes[node].get('state')}, degree={G.degree(node)}"
        )

    print("\n--- FINAL EVALUATION ---")
    final_mean, x_final = evaluate_seed_set(seeds, W, s, alpha)
    baseline_mean = np.mean(x_baseline)
    print(f"Baseline mean opinion : {baseline_mean:.4f}")
    print(f"Final mean opinion    : {final_mean:.4f}")
    print(f"Improvement           : {final_mean - baseline_mean:.4f}")
    print(f"Users improved     : {np.mean(x_final > x_baseline):.4f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, f"{year}_results.pickle"), "wb") as f:
        pickle.dump(
            {
                "seeds": seeds,
                "x_baseline": x_baseline,
                "scores": scores,
                "candidate_indices": candidate_indices,
                "x_final": x_final,
            },
            f,
            pickle.HIGHEST_PROTOCOL,
        )

    plot_distribution_shift(x_baseline, x_final, year)
    plot_choropleth(G, nodes, x_baseline, x_final, year)
