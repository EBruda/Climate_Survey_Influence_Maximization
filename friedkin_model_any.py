import networkx as nx
import numpy as np
import pickle
import scipy.sparse as sp
import pandas as pd
import plotly.express as px
from urllib.request import urlopen
import json
import time
import matplotlib.pyplot as plt
year = "2019"


def load_graph_and_matrices(filepath):
    print("Loading graph...")
    with open(filepath, 'rb') as f:
        G = pickle.load(f)

    nodes = list(G.nodes())
    n = len(nodes)
    print(f"Graph loaded with {n} nodes.")

    # Baseline opinions
    s = np.array([G.nodes[node]['baseline_opinion'] for node in nodes])

    # Susceptibility
    alpha = np.array([G.nodes[node]['susceptibility'] for node in nodes])

    print("Building sparse weight matrix (W)...")
    W_raw = nx.to_scipy_sparse_array(G, nodelist=nodes, weight='weight', format='csr')

    # Row normalize
    row_sums = np.array(W_raw.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0

    D_inv = sp.diags(1.0 / row_sums)
    W = D_inv @ W_raw

    return G, nodes, W, s, alpha


# =========================
# FJ SIMULATION (ITERATIVE)
# =========================
def run_fj_simulation(W, s, alpha, max_iters=100, tolerance=1e-5):
    x = np.copy(s)

    for i in range(max_iters):
        x_new = alpha * (W @ x) + (1.0 - alpha) * s

        if np.linalg.norm(x_new - x, ord=1) < tolerance:
            break

        x = x_new

    return x


# =========================
# INFLUENCE SCORE (FAST)
# =========================
def compute_influence_scores(W, alpha, steps=20):
    """
    Power-iteration style approximation of influence
    """
    print("Computing influence scores...")
    n = W.shape[0]

    v = np.ones(n)
    scores = np.zeros(n)

    for _ in range(steps):
        v = alpha * (W @ v)
        scores += v

    return scores


# =========================
# CANDIDATE SELECTION
# =========================
def get_top_candidates(scores, top_k=1000):
    print(f"Selecting top {top_k} candidate nodes...")
    idx_sorted = np.argsort(scores)[::-1]
    return idx_sorted[:top_k]


# =========================
# GREEDY INFLUENCE MAX
# =========================
def greedy_influence_max(W, s, alpha, candidate_indices, k=10, max_iters=50):
    print("Running scalable greedy influence maximization...")

    # Baseline
    base_x = run_fj_simulation(W, s, alpha)
    base_score = np.mean(base_x)

    selected = []

    for step in range(k):
        best_gain = -np.inf
        best_node = None

        print(f"\nStep {step+1}/{k}")

        for idx in candidate_indices:
            if idx in selected:
                continue

            # Copy state
            temp_s = np.copy(s)
            temp_alpha = np.copy(alpha)

            # Apply intervention
            temp_s[idx] = 1.0
            temp_alpha[idx] = 0.0

            x_new = run_fj_simulation(W, temp_s, temp_alpha, max_iters=max_iters)
            score = np.mean(x_new)

            gain = score - base_score

            if gain > best_gain:
                best_gain = gain
                best_node = idx

        selected.append(best_node)
        base_score += best_gain

        print(f"Selected node index: {best_node} | Gain: {best_gain:.6f}")

    return selected


# =========================
# EVALUATION FUNCTION
# =========================
def evaluate_seed_set(seed_indices, W, s, alpha):
    temp_s = np.copy(s)
    temp_alpha = np.copy(alpha)

    for idx in seed_indices:
        temp_s[idx] = 1.0
        temp_alpha[idx] = 0.0

    final_x = run_fj_simulation(W, temp_s, temp_alpha)

    return np.mean(final_x), final_x



def plot_distribution_shift(x_baseline, x_final):
    plt.figure()

    plt.hist(x_final, bins=50, alpha=0.5, label="After Seeding")
    plt.hist(x_baseline, bins=50, alpha=0.5, label="Baseline")

    plt.xlabel("Opinion Value")
    plt.ylabel("Number of Counties")
    plt.title(f"Opinion Distribution Shift - {year}")
    plt.legend()
    plt.savefig(f"visualizations/{year}_opinion_distribution_shift.png")
    # plt.show()


if __name__ == "__main__":
    G, nodes, W, s, alpha = load_graph_and_matrices(f'network_graphs/network_graph_fj_{year}.pickle')

    print("\n--- BASELINE SIMULATION ---")
    start = time.time()
    x_baseline = run_fj_simulation(W, s, alpha)
    print(f"Baseline mean opinion: {np.mean(x_baseline):.4f}")
    print(f"Time: {time.time() - start:.2f}s")

    print("\n--- COMPUTING INFLUENCE SCORES ---")
    start = time.time()
    scores = compute_influence_scores(W, alpha, steps=20)
    print(f"Time: {time.time() - start:.2f}s")

    print("\n--- SELECTING CANDIDATES ---")
    candidate_indices = get_top_candidates(scores, top_k=1000)

    print("\n--- RUNNING GREEDY INFLUENCE MAX ---")
    start = time.time()
    seeds = greedy_influence_max(W, s, alpha, candidate_indices, k=10)
    print(f"Time: {time.time() - start:.2f}s")

    print("\n--- FINAL RESULTS ---")
    seed_nodes = [nodes[i] for i in seeds]
    # print("Selected seed indices:", seeds)
    # print("Selected seed nodes:", seed_nodes)

    print("\n--- FINAL EVALUATION ---")
    final_mean, _ = evaluate_seed_set(seeds, W, s, alpha)
    print(f"Final mean opinion after seeding: {final_mean:.4f}")

    print("\n--- SEED NODE DETAILS ---")
    for seed_idx in seeds:
        node = nodes[seed_idx]
        degree = G.degree(node)
        county = G.nodes[node]['county']
        state = G.nodes[node]['state']
        print(f"Node {seed_idx}: {county}, {state}, Degree: {degree}")
    
    print("\n--- FINAL EVALUATION ---")
    final_mean, x_final = evaluate_seed_set(seeds, W, s, alpha)
    baseline_mean = np.mean(x_baseline)
    print(f"Baseline mean opinion: {baseline_mean:.4f}")
    print(f"Final mean opinion after seeding: {final_mean:.4f}")
    print(f"Improvement: {final_mean - baseline_mean:.4f}")
    print(f"Fraction of counties with increased opinion: {np.mean(x_final > x_baseline):.4f}")

    results = {
        "seeds": seeds,
        "x_baseline": x_baseline,
        "scores": scores,
        "candidiate_indices": candidate_indices, 
        "x_final": x_final
    }

    with open(f'fj_model_results/{year}_results.pickle', 'wb') as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

    plot_distribution_shift(x_baseline, x_final)


    delta_x = (x_final - x_baseline) / 100.0
    
    # 1. Extract the data (ignoring FIPS here, we will merge it next)
    data = []
    for i, node in enumerate(nodes):
        data.append({
            'GeoName': G.nodes[node].get('GeoName', ''),
            'County': G.nodes[node].get('county', ''),
            'State': G.nodes[node].get('state', ''),
            'Opinion_Delta': delta_x[i]
        })
    
    df_raw = pd.DataFrame(data)
    
    # 2. Aggregate! Group by State and County ONLY.
    df_county = df_raw.groupby(['State', 'County'])['Opinion_Delta'].mean().reset_index()
    
    target_topic = "worried"
    target_oppose = target_topic + "Oppose"


    yale_lookup = pd.read_excel("YCOM_2024_publicdata.xlsx", sheet_name='YCOM_2010_2024')
    
    # Clean it up to get a pure State + County -> GeoID mapping
    yale_lookup = yale_lookup[yale_lookup['GeoType'] == 'county'].copy()
    yale_lookup[['County', 'State']] = yale_lookup['GeoName'].str.split(',', expand=True)
    yale_lookup['County'] = yale_lookup['County'].str.strip()
    yale_lookup['State'] = yale_lookup['State'].str.strip()
    # yale_lookup = yale_lookup[(yale_lookup["worried"] == target_topic) | (yale_lookup["worried"] == target_oppose)].copy()
    yale_lookup = yale_lookup[yale_lookup[f"x{year}"].notna()].copy()
    
    df_county = df_raw.groupby(['State', 'County'])['Opinion_Delta'].mean().reset_index()
   
   # 2.5 LOAD FIPS FROM THE ORIGINAL YALE DATASET
   # Load the raw yale data just to act as a FIPS dictionary
    yale_lookup = pd.read_excel("YCOM_2024_publicdata.xlsx", sheet_name='YCOM_2010_2024')
   
   # Clean it up to get a pure State + County -> GeoID mapping
    yale_lookup = yale_lookup[yale_lookup['GeoType'] == 'county'].copy()
    yale_lookup[['County', 'State']] = yale_lookup['GeoName'].str.split(',', expand=True)
    yale_lookup['County'] = yale_lookup['County'].str.strip()
    yale_lookup['State'] = yale_lookup['State'].str.strip()
    
    # Merge the new GeoID into your aggregated dataframe
    df_county = df_county.merge(yale_lookup[['County', 'State', 'GeoID']], on=['County', 'State'], how='left')
    
    # Drop any rows where Yale couldn't find a matching GeoID
    df_county = df_county.dropna(subset=['GeoID']).copy()
    
    # Convert to integer (removes decimals), then to string, then pad to 5 digits
    df_county['FIPS'] = df_county['GeoID'].astype(int).astype(str).str.zfill(5)
    
    # Drop the now-redundant GeoID column
    df_county = df_county.drop(columns=['GeoID'])
    
    # 3. Load Plotly GeoJSON
    with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
        counties = json.load(response)
    
    # 4. Plot
    fig = px.choropleth(df_county, 
                        geojson=counties, 
                        locations='FIPS', 
                        color='Opinion_Delta',
                        color_continuous_scale="RdBu", 
                        color_continuous_midpoint=0,   
                        scope="usa",
                        hover_name="County",
                        hover_data=["State", "Opinion_Delta"],
                        title=f"Change in Opinion after Seeding (Delta) - {year}")
    fig.write_image(f"visualizations/change_opinion_{year}.png")
    # fig.show()
    