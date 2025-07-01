import networkx as nx
import matplotlib.pyplot as plt
import random

# random seed for reproducibility
random.seed(42)

# Parameters
N = 1000                    # Number of nodes
m = 2                      # Edges per new node in BA model
phi = 0.5                  # Threshold for adoption
max_rounds = 50            # Max rounds for dynamic simulation

def identify_influencers_by_threshold(G):
    """Identify influencers as nodes whose degree exceeds twice the average."""
    avg_deg = sum(dict(G.degree()).values()) / G.number_of_nodes()
    threshold = 2 * avg_deg
    influencers = {n for n, d in G.degree() if d > threshold}
    print(f"Average degree = {avg_deg:.2f}, threshold = {threshold:.2f}")
    print(f"Number of influencers: {len(influencers)}")
    return influencers

def static_majority_illusion(G, opinions):
    """Compute nodes under strict majority illusion."""
    total_red = sum(1 for v in opinions.values() if v == 'Red')
    total_blue = len(opinions) - total_red
    global_maj = 'Red' if total_red > total_blue else 'Blue'
    illusion_nodes = []
    for v in G.nodes():
        nbrs = list(G.neighbors(v))
        if not nbrs:
            continue
        local_red = sum(opinions[u] == 'Red' for u in nbrs)
        local_blue = len(nbrs) - local_red
        if local_red == local_blue:
            continue
        local_maj = 'Red' if local_red > local_blue else 'Blue'
        if local_maj != global_maj:
            illusion_nodes.append(v)
    return global_maj, illusion_nodes

def plot_static_illusion(G, opinions, illusion_nodes):
    """Static network plot with illusioned nodes."""
    pos = nx.spring_layout(G, seed=42)
    colors = ['red' if opinions[v]=='Red' else 'skyblue' for v in G]
    shapes = {v: ('s' if v in illusion_nodes else 'o') for v in G}
    
    plt.figure(figsize=(8,6))
    for shape in ['o','s']:
        nodes = [v for v in G if shapes[v]==shape]
        nx.draw_networkx_nodes(
            G, pos, nodelist=nodes,
            node_color=[colors[v] for v in nodes],
            node_shape=shape, edgecolors='black', linewidths=1.0, node_size=300
        )
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.title(f"Static Illusion (n_influencers={sum(1 for v in opinions if opinions[v]=='Red')})")
    plt.axis('off')
    plt.show()

def dynamic_simulation(G, opinions_init):
    """Run threshold diffusion, track illusion counts, and return final opinions."""
    opinions = opinions_init.copy()
    illusion_series = []
    for _ in range(max_rounds):
        _, illusion_nodes = static_majority_illusion(G, opinions)
        illusion_series.append(len(illusion_nodes))
        # update opinions
        new_op = opinions.copy()
        for v in G.nodes():
            if opinions[v] == 'Blue':
                red_nbrs = sum(opinions[u]=='Red' for u in G.neighbors(v))
                if red_nbrs > phi * G.degree[v]:
                    new_op[v] = 'Red'
        if new_op == opinions:
            break
        opinions = new_op
    return illusion_series, opinions

def plot_illusion_development(illusion_series):
    """Plot the number of illusioned nodes over time."""
    plt.figure(figsize=(6,4))
    plt.plot(range(len(illusion_series)), illusion_series, marker='s')
    plt.title("Development of Majority Illusion Over Time")
    plt.xlabel("Round")
    plt.ylabel("Number of Nodes Under Illusion")
    plt.grid(True)
    plt.show()

def plot_network_evolution(G, opinions_start, opinions_end):
    """Show initial vs final network coloring."""
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(12,5))
    
    # Initial
    plt.subplot(1,2,1)
    colors_start = ['red' if opinions_start[v]=='Red' else 'skyblue' for v in G]
    nx.draw(G, pos, node_color=colors_start, with_labels=False, node_size=300, edge_color='gray')
    plt.title("t = 0 (Initial Opinions)")
    plt.axis('off')
    
    # Final
    plt.subplot(1,2,2)
    colors_end = ['red' if opinions_end[v]=='Red' else 'skyblue' for v in G]
    nx.draw(G, pos, node_color=colors_end, with_labels=False, node_size=300, edge_color='gray')
    plt.title("t = final (Post-Diffusion)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Build network once
    G = nx.barabasi_albert_graph(N, m, seed=42)
    
    # Identify influencers
    influencers = identify_influencers_by_threshold(G)
    
    # Static
    opinions_init = {v: ('Red' if v in influencers else 'Blue') for v in G}
    gm, illusion_nodes = static_majority_illusion(G, opinions_init)
    print(f"Global majority: {gm}, Illusioned nodes: {len(illusion_nodes)}")
    plot_static_illusion(G, opinions_init, illusion_nodes)
    
    # Dynamic
    illusion_series, opinions_final = dynamic_simulation(G, opinions_init)
    print(f"Final number under illusion: {illusion_series[-1]}")
    plot_illusion_development(illusion_series)
    
    # Network evolution
    plot_network_evolution(G, opinions_init, opinions_final)
