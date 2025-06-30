import networkx as nx
import matplotlib.pyplot as plt
import random
from statistics import mean, stdev

# Ensure reproducibility
random.seed(42)

# Parameters
N = 1000                  # Total nodes
m = 2                    # BA model parameter (edges per new node)
phi = 0.5                # Majority threshold
max_rounds = 50          # Max dynamic rounds
minority_fracs = [0.10, 0.30, 0.40]  # Tested minority ratios

def identify_influencers_by_threshold(G):
    """Return set of nodes with degree > 2 × average degree."""
    avg_deg = sum(dict(G.degree()).values()) / G.number_of_nodes()
    threshold = 2 * avg_deg
    return {v for v, d in G.degree() if d > threshold}

def static_majority_illusion(G, opinions):
    """Return (global_majority, list of nodes under strict majority illusion)."""
    red = sum(op == 'Red' for op in opinions.values())
    blue = len(opinions) - red
    global_maj = 'Red' if red > blue else 'Blue'
    illusion = []
    for v in G:
        nbrs = list(G.neighbors(v))
        if not nbrs:
            continue
        r = sum(opinions[u] == 'Red' for u in nbrs)
        b = len(nbrs) - r
        if r == b:
            continue
        local = 'Red' if r > b else 'Blue'
        if local != global_maj:
            illusion.append(v)
    return global_maj, illusion

def dynamic_simulation(G, opinions_init):
    """Run reversible majority-vote dynamics; return illusion count series and final opinions."""
    opinions = opinions_init.copy()
    illusion_series = []
    for _ in range(max_rounds):
        _, ill = static_majority_illusion(G, opinions)
        illusion_series.append(len(ill))
        new_op = {}
        for v in G:
            nbrs = list(G.neighbors(v))
            if not nbrs:
                new_op[v] = opinions[v]
                continue
            r = sum(opinions[u] == 'Red' for u in nbrs)
            b = len(nbrs) - r
            if r > b:
                new_op[v] = 'Red'
            elif b > r:
                new_op[v] = 'Blue'
            else:
                new_op[v] = opinions[v]
        if new_op == opinions:
            break
        opinions = new_op
    return illusion_series, opinions

def plot_static_illusion_detailed(G, opinions, influencers, extra, illusion_nodes, title):
    """Plot static network, distinguishing influencers, extra minority, and majority."""
    pos = nx.spring_layout(G, seed=42)
    # categorize nodes
    infl = list(influencers)
    extra_only = [v for v in extra if v not in influencers]
    majority = [v for v in G if v not in influencers and v not in extra]
    plt.figure(figsize=(8,6))
    # influencers: red triangles
    nx.draw_networkx_nodes(
        G, pos, nodelist=infl,
        node_color='red', node_shape='^', node_size=400,
        label='Influencers',
        edgecolors=['black' if v in illusion_nodes else 'none' for v in infl],
        linewidths=2
    )
    # extra minority: pink squares
    nx.draw_networkx_nodes(
        G, pos, nodelist=extra_only,
        node_color='pink', node_shape='s', node_size=300,
        label='Minority (extra)',
        edgecolors=['black' if v in illusion_nodes else 'none' for v in extra_only],
        linewidths=2
    )
    # majority: skyblue circles
    nx.draw_networkx_nodes(
        G, pos, nodelist=majority,
        node_color='skyblue', node_shape='o', node_size=300,
        label='Majority',
        edgecolors=['black' if v in illusion_nodes else 'none' for v in majority],
        linewidths=2
    )
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    plt.title(title)
    plt.legend(scatterpoints=1, fontsize=10)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_illusion_development(series, title):
    """Plot the number under illusion over each round."""
    plt.figure(figsize=(5,4))
    plt.plot(range(len(series)), series, marker='s')
    plt.title(title)
    plt.xlabel('Round')
    plt.ylabel('Nodes Under Illusion')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_network_evolution(G, start, end, title):
    """Side by side: initial vs. final network coloring."""
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    cols0 = ['red' if start[v]=='Red' else 'skyblue' for v in G]
    nx.draw(G, pos, node_color=cols0, node_size=200, edge_color='gray', with_labels=False)
    plt.title('t = 0'); plt.axis('off')
    plt.subplot(1,2,2)
    cols1 = ['red' if end[v]=='Red' else 'skyblue' for v in G]
    nx.draw(G, pos, node_color=cols1, node_size=200, edge_color='gray', with_labels=False)
    plt.title('t = final'); plt.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Single-run scenarios with random BA networks
    for frac in minority_fracs:
        G = nx.barabasi_albert_graph(N, m, seed=random.randrange(10000))
        influencers = identify_influencers_by_threshold(G)
        target = int(frac * N)

        # Cap influencer set if too large
        if len(influencers) > target:
            # keep top-'target' hubs by degree
            sorted_infl = sorted(influencers, key=lambda v: G.degree(v), reverse=True)
            minority_set = set(sorted_infl[:target])
        else:
            # keep all influencers, then add random extras
            minority_set = set(influencers)
            rest = list(set(G.nodes()) - minority_set)
            extra = target - len(minority_set)
            if extra > 0:
                minority_set |= set(random.sample(rest, extra))

        opinions_init = {v: 'Red' if v in minority_set else 'Blue' for v in G}
        gm, illusion_nodes = static_majority_illusion(G, opinions_init)
        title = f"Static Illusion: {int(frac*100)}% minority, {len(influencers)} influencers"
        plot_static_illusion_detailed(
            G, opinions_init, influencers, minority_set - influencers,
            illusion_nodes, title
        )

        ill_series, final_op = dynamic_simulation(G, opinions_init)
        print(f"{int(frac*100)}% | infl={len(influencers)} | global maj={gm}")
        print(f"  static={len(illusion_nodes)} | peak={max(ill_series)} | final={ill_series[-1]}")

        plot_illusion_development(
            ill_series,
            f"Illusion Over Time: {int(frac*100)}% / infl={len(influencers)}"
        )
        plot_network_evolution(
            G, opinions_init, final_op,
            f"Network Evolution: {int(frac*100)}% / infl={len(influencers)}"
        )

    # Batch simulations recording influencer count
    runs = 200
    records = []
    for run in range(runs):
        G_run = nx.barabasi_albert_graph(N, m, seed=random.randrange(10000))
        infl_run = identify_influencers_by_threshold(G_run)
        icount = len(infl_run)
        for frac in minority_fracs:
            target = int(frac * N)
            # cap or fill minority as above
            if icount > target:
                sorted_infl = sorted(infl_run, key=lambda v: G_run.degree(v), reverse=True)
                minority = set(sorted_infl[:target])
            else:
                minority = set(infl_run)
                rest = list(set(G_run.nodes()) - minority)
                extra = target - len(minority)
                if extra > 0:
                    minority |= set(random.sample(rest, extra))
            opinions = {v: 'Red' if v in minority else 'Blue' for v in G_run}
            static_ct = len(static_majority_illusion(G_run, opinions)[1])
            seq, _ = dynamic_simulation(G_run, opinions)
            records.append({
                'infl': icount,
                'frac': frac,
                'static': static_ct,
                'final': seq[-1]
            })

    # Summarize by influencer count and minority fraction
    infl_counts = sorted(set(r['infl'] for r in records))
    print("\nInfl | 10% stat 10% fin | 30% stat 30% fin | 40% stat 40% fin")
    print("-"*70)
    for ic in infl_counts:
        row = [f"{ic:>4d}"]
        for frac in minority_fracs:
            subset = [r for r in records if r['infl']==ic and r['frac']==frac]
            if subset:
                s_vals = [r['static'] for r in subset]
                f_vals = [r['final']  for r in subset]
                s_mean, s_sd = mean(s_vals), stdev(s_vals) if len(s_vals)>1 else 0
                f_mean, f_sd = mean(f_vals), stdev(f_vals) if len(f_vals)>1 else 0
                row.append(f"{s_mean:6.1f}±{s_sd:<5.1f}")
                row.append(f"{f_mean:6.1f}±{f_sd:<5.1f}")
            else:
                row.extend(["   -    ", "   -    "])
        print(" | ".join(row))
