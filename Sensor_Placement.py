import networkx as nx 
import numpy as np
import time
import random
from collections import defaultdict
from typing import Set, Dict, Tuple, Callable, List

def graph(num_nodes: int,p: float) -> nx.Graph:
    G = nx.fast_gnp_random_graph(num_nodes, p, seed=42)
    if not nx.is_connected(G):
        G = nx.Graph(G)
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        while G.number_of_nodes() < num_nodes*0.95:
            node_to_add = set(range(num_nodes)) - set(G.nodes())
            if not node_to_add:
                break
            new_node = node_to_add.pop()
            G.add_node(new_node)
            existing_node = random.choice(list(G.nodes()))
            G.add_edge(new_node, existing_node)

    mapping = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    return G

def greedy_2_approximation(G:nx.Graph)->Set[int]:
    V_prime = set()
    E_prime = set(G.edges())

    while E_prime:
        u,v = next(iter(E_prime))
        V_prime.add(u)
        V_prime.add(v)

        E_prime = {e for e in E_prime if u not in e and v not in e}

    return V_prime

def heuristic_vc(G: nx.Graph)->Set[int]:
    V_prime = set()
    H = G.copy()

    while H.number_of_edges()>0:
        degrees = dict(H.degree())

        if not degrees:
            break

        v_star = max(degrees, key=degrees.get)
        V_prime.add(v_star)
        H.remove_node(v_star)

    return V_prime

def rand_heuristic_vc(G: nx.Graph, max_iterations: int=100)->Set[int]:
    best_V_prime = set(G.nodes())
    for _ in range(max_iterations):
        V_prime = set()
        H = G.copy()
        while H.number_of_edges()>0:
            nodes_with_edges = [n for n,d in H.degree() if d>0]
            if not nodes_with_edges:
                break

            weights = np.array([H.degree(n) for n in nodes_with_edges])
            probabilities = weights / np.sum(weights)

            v_star = np.random.choice(nodes_with_edges, p=probabilities)

            V_prime.add(v_star)
            H.remove_node(v_star)
        if len(V_prime)<len(best_V_prime):
            best_V_prime = V_prime
    return best_V_prime

def verify_cover(G: nx.Graph, V_prime: Set[int]) -> bool:
    return all(u in V_prime or v in V_prime for u, v in G.edges())  

def compare_algorithms(G: nx.Graph, num_runs: int = 5) -> Dict:
    algorithms: Dict[str, Callable[[nx.Graph], Set[int]]] = {
        "Greedy 2-Approx": greedy_2_approximation,
        "Max-Degree Heuristic (LP-surrogate)": heuristic_vc,
        "Randomized Heuristic": rand_heuristic_vc,
    }
    results = defaultdict(lambda: defaultdict(list))
    for name, func in algorithms.items():
        for _ in range(num_runs if "Randomized" in name else 1):  # Run randomized multiple times
            start = time.time()
            V_prime = func(G)
            end = time.time()

            exec_time = end - start
            size = len(V_prime)
            coverage = verify_cover(G, V_prime)

            results[name]['time'].append(exec_time)
            results[name]['size'].append(size)
            results[name]['coverage'].append(coverage)

    final_results = {}
    for name, res in results.items():
        final_results[name] = {
            "Num Selected Sensors (Avg)": np.mean(res['size']),
            "Execution Time (Avg, s)": np.mean(res['time']),
            "Coverage Completeness": all(res['coverage'])  # Should be True for all VC algorithms
        }

    return final_results

N = 750
P = 0.01
Iot_Graph = graph(N,P)

print("--- IoT Sensor Network Graph Details ---")
print(f"Nodes (Sensors): {Iot_Graph.number_of_nodes()}")
print(f"Edges (Links): {Iot_Graph.number_of_edges()}")
print("-" * 40)
comparison_results = compare_algorithms(Iot_Graph, num_runs=25)
print("\n--- Sensor Placement Algorithm Comparison ---")
print("{:<40} | {:<25} | {:<20} | {:<20}".format(
    "Algorithm",
    "Avg # Selected Sensors",
    "Avg Execution Time (s)",
    "Coverage Completeness"
))
print("-" * 110)

for name, metrics in comparison_results.items():
    print("{:<40} | {:<25.2f} | {:<20.6f} | {:<20}".format(
        name,
        metrics["Num Selected Sensors (Avg)"],
        metrics["Execution Time (Avg, s)"],
        "Complete" if metrics["Coverage Completeness"] else "Incomplete"
    ))
print("-" * 110)

print("\n--- Discussion of Trade-offs ---")
print("1. Greedy 2-Approximation: Fast and guarantees a 2-approximation ratio, but may select more nodes than necessary.")
print("2. Max-Degree Heuristic: Often performs well in practice, faster than exact methods, but no guaranteed approximation ratio.")
print("3. Randomized Heuristic: Can find smaller covers in some cases due to randomization, but results vary and may require multiple runs.")
print("\n--- Suggested Improvements ---")
print("1. Use exact algorithms like dynamic programming for small graphs.")
print("2. Integrate with LP solvers (e.g., PuLP) for true LP-relaxation.")
print("3. Parallelize randomized runs for better average performance.")
print("4. Consider graph properties (e.g., degree distribution) for algorithm selection.")

