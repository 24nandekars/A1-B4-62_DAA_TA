# Randomized and Probabilistic Vertex Cover Algorithms
import networkx as nx
import random
import time
import matplotlib.pyplot as plt
import statistics
import multiprocessing as mp
from multiprocessing import Pool

def randomized_vertex_cover(G):
    VC = set()
    edges = list(G.edges())
    while edges:
        edge = random.choice(edges)
        u, v = edge
        chosen = random.choice([u, v])
        VC.add(chosen)
        edges = [e for e in edges if chosen not in e]
    return VC

def greedy_vertex_cover(G):
    VC = set()
    G_copy = G.copy()
    while G_copy.edges():
        degrees = dict(G_copy.degree())
        v = max(degrees, key=degrees.get)
        VC.add(v)
        G_copy.remove_node(v)
    return VC

def run_trial(args):
    n, p, algorithm = args
    G = nx.erdos_renyi_graph(n, p)
    start = time.time()
    vc = algorithm(G)
    end = time.time()
    return len(vc), end - start

def main():
    n = 100
    trials = 100
    densities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    avg_sizes_rand = []
    variances_rand = []
    avg_times_rand = []
    avg_sizes_greedy = []
    avg_times_greedy = []

    for p in densities:
        print(f"Running for density p={p}")

        with Pool(mp.cpu_count()) as pool:
            results_rand = pool.map(run_trial, [(n, p, randomized_vertex_cover)] * trials)
        sizes_rand = [r[0] for r in results_rand]
        times_rand = [r[1] for r in results_rand]
        avg_sizes_rand.append(statistics.mean(sizes_rand))
        variances_rand.append(statistics.variance(sizes_rand))
        avg_times_rand.append(statistics.mean(times_rand))

        with Pool(mp.cpu_count()) as pool:
            results_greedy = pool.map(run_trial, [(n, p, greedy_vertex_cover)] * trials)
        sizes_greedy = [r[0] for r in results_greedy]
        times_greedy = [r[1] for r in results_greedy]
        avg_sizes_greedy.append(statistics.mean(sizes_greedy))
        avg_times_greedy.append(statistics.mean(times_greedy))

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    axs[0,0].plot(densities, avg_sizes_rand, label='Randomized', marker='o')
    axs[0,0].plot(densities, avg_sizes_greedy, label='Greedy', marker='s')
    axs[0,0].set_xlabel('Graph Density (p)')
    axs[0,0].set_ylabel('Average Vertex Cover Size')
    axs[0,0].set_title('Average VC Size vs Density')
    axs[0,0].legend()
    axs[0,0].grid(True)

    axs[0,1].plot(densities, variances_rand, label='Randomized', marker='o')
    axs[0,1].set_xlabel('Graph Density (p)')
    axs[0,1].set_ylabel('Variance of VC Size')
    axs[0,1].set_title('Variance of VC Size vs Density')
    axs[0,1].legend()
    axs[0,1].grid(True)

    axs[1,0].plot(densities, avg_times_rand, label='Randomized', marker='o')
    axs[1,0].plot(densities, avg_times_greedy, label='Greedy', marker='s')
    axs[1,0].set_xlabel('Graph Density (p)')
    axs[1,0].set_ylabel('Average Execution Time (s)')
    axs[1,0].set_title('Execution Time vs Density')
    axs[1,0].legend()
    axs[1,0].grid(True)

    ratios = [r / g for r, g in zip(avg_sizes_rand, avg_sizes_greedy)]
    axs[1,1].plot(densities, ratios, marker='o')
    axs[1,1].set_xlabel('Graph Density (p)')
    axs[1,1].set_ylabel('Approx Ratio (Rand / Greedy)')
    axs[1,1].set_title('Approximation Ratio vs Density')
    axs[1,1].grid(True)

    plt.tight_layout()
    plt.savefig('vc_analysis.png')
    plt.show()

    print("\nTesting on large graph (n=1000, p=0.1)")
    n_large = 1000
    p_large = 0.1
    trials_large = 10

    with Pool(mp.cpu_count()) as pool:
        results_large = pool.map(run_trial, [(n_large, p_large, randomized_vertex_cover)] * trials_large)
    sizes_large = [r[0] for r in results_large]
    times_large = [r[1] for r in results_large]
    print(f"Average VC size: {statistics.mean(sizes_large)}")
    print(f"Average time: {statistics.mean(times_large)} s")

if __name__ == "__main__":
    main()
