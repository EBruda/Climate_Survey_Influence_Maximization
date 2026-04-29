# Climate Survey Influence Maximization

This project applies the Friedkin-Johnsen (FJ) opinion dynamics model to U.S. county-level climate opinion data, using the Gowalla social network to identify the most influential nodes for shifting public opinion on climate change.

## Data Source Set up

- **Yale Climate Opinion Maps (YCOM)** — `YCOM_2024_publicdata.xlsx`  
  County-level survey data on climate opinions (2010–2024). Download from [Yale Program on Climate Change Communication](https://climatecommunication.yale.edu/visualizations-data/ycom-us/) after signing up for it. Place the `YCOM_2024_publicdata.xlsx` file in the root directory. 

- **Gowalla** — `gowalla_data/`  
  A location-based social network with check-in data used to construct a friendship graph of U.S. users. Download from [SNAP](https://snap.stanford.edu/data/loc-Gowalla.html). Download the `loc-gowalla_edges.txt.gz` and `loc-gowalla_totalCheckins.txt.gz	` files. Unzip both and place their unzipped files in the `gowalla_data` folder in the root directory. 

  After setting up both data sources, your root directory should contain:
  ```
  gowalla_data/
    - Gowalla_edges.txt
    - Gowalla_totalCheckins.txt

  YCOM_2024_publicdata.xlsx
  ```

## Environment Setup 
Create the environment by running `conda env create -f environment.yml`

Then activate the environment using `conda activate climate-env`

## Main Scripts

### [`run_fj.py`](run_fj.py)
The core pipeline. Run it for a single year via:
```bash
python run_fj.py --year 2022
```

It runs four stages in sequence:

1. **Build network graph** — Loads YCOM county opinions and Gowalla check-ins, reverse-geocodes users to their home county, and constructs a weighted `networkx` graph where each node holds a baseline opinion derived from the YCOM "worried/worriedOppose" survey question. 
This uses the cached network graphs found at `network_graphs/network_graph_fj_{year}.pickle`. If you would like to build the network graph for that year from scratch, then run the `create_network_graph.ipynb` notebook first.  

2. **Friedkin-Johnsen simulation** — Runs the Friedkin-Johnsen iterative update (`x = α·Wx + (1−α)·s`) to find the equilibrium opinion vector.

3. **Greedy influence maximization** — Scores all nodes by influence (power-iteration proxy), selects the top-1000 candidates, then greedily picks the 10 seeds whose intervention (pinning a node to full support) maximizes the mean opinion lift.

4. **Visualization & results** — Saves outputs to `fixed_fj_visualizations/` and `fj_model_results/`.

**Key configuration constants** (top of file):
| Constant | Default | Description |
|---|---|---|
| `YEAR` | `"2020"` | Overridden by `--year` flag |
| `TARGET_TOPIC` | `"worried"` | YCOM survey column |
| `EXCEL_PATH` | `YCOM_2024_publicdata.xlsx` | YCOM data file |

### [`run_all_years.sh`](run_all_years.sh)
Runs `fixed_fj.py` for every year from 2018 to 2024 in sequence:
```bash
bash run_all_years.sh
```
Requires `PYTHON` to point to a valid Python interpreter (default: `/opt/anaconda3/bin/python3`).

### [`utils/utils.py`](utils/utils.py)
Lookup tables mapping Connecticut town names to their historical counties (pre-2023) and planning regions (2023+), used during Gowalla pre-processing to handle Connecticut's county boundary changes.

## Outputs

| Directory | Contents |
|---|---|
| `network_graphs/` | Cached `networkx` graphs per year (`.pickle`) |
| `fj_model_results/` | Seed sets, baseline/final opinion vectors, influence scores per year (`.pickle`) |
| `fixed_fj_visualizations/` | Per-year choropleth maps (`.html`) and opinion distribution shift plots (`.png`) |

## Dependencies

```
networkx, numpy, pandas, scipy, matplotlib, plotly, tqdm, reverse_geocoder, openpyxl
```

Install with:
```bash
pip install networkx numpy pandas scipy matplotlib plotly tqdm reverse_geocoder openpyxl
```
