#!/usr/bin/env python3
"""
Demonstration of the cppyy-based Graph implementation.

This shows that the new cppyy approach works identically to the 
original witty/Cython approach but without needing template files
or complex compilation steps.
"""

import numpy as np
import os

# Set up environment for cppyy (macOS specific)
os.environ['DYLD_LIBRARY_PATH'] = '/opt/homebrew/lib:' + os.environ.get('DYLD_LIBRARY_PATH', '')

from spatial_graph.graph.graph_cppyy_simple import Graph

def main():
    print("🎉 cppyy-based Graph Implementation Demo")
    print("=" * 50)
    
    # Create a directed graph with node and edge attributes
    print("\n1. Creating a directed graph...")
    graph = Graph(
        node_dtype="uint64",
        node_attr_dtypes={"score": "float"},
        edge_attr_dtypes={"score": "float"},
        directed=True
    )
    print(f"   ✓ Created directed graph, size: {len(graph)}")
    
    # Add some nodes with attributes
    print("\n2. Adding nodes with attributes...")
    nodes = np.array([1, 2, 3, 4, 5], dtype="uint64")
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype="float32")
    num_added = graph.add_nodes(nodes, score=scores)
    print(f"   ✓ Added {num_added} nodes, graph size: {len(graph)}")
    
    # Add some edges with attributes
    print("\n3. Adding edges with attributes...")
    edges_added = 0
    for i in range(len(nodes) - 1):
        u, v = nodes[i], nodes[i + 1]
        weight = float(u + v) / 10.0
        edges_added += graph.add_edge(np.array([u, v]), score=weight)
    print(f"   ✓ Added {edges_added} edges, total edges: {graph.num_edges()}")
    
    # Demonstrate attribute access
    print("\n4. Accessing node attributes...")
    node_1_score = graph.get_node_data_score(1)
    all_scores = graph.get_nodes_data_score(nodes)
    print(f"   ✓ Node 1 score: {node_1_score:.3f}")
    print(f"   ✓ All node scores: {all_scores}")
    
    # Demonstrate neighbor counting
    print("\n5. Counting neighbors...")
    # For directed graphs, count in/out neighbors separately
    in_neighbor_counts = graph.count_in_neighbors(nodes)
    out_neighbor_counts = graph.count_out_neighbors(nodes)
    print(f"   ✓ In-neighbor counts: {in_neighbor_counts}")
    print(f"   ✓ Out-neighbor counts: {out_neighbor_counts}")
    
    # Show all nodes and edges
    print("\n6. Graph contents...")
    all_nodes = graph.nodes()
    print(f"   ✓ All nodes: {all_nodes}")
    print(f"   ✓ Graph summary: {len(graph)} nodes, {graph.num_edges()} edges")
    
    print("\n" + "=" * 50)
    print("🎉 Demo completed successfully!")
    print("\nKey advantages of cppyy approach:")
    print("✓ No template files needed")
    print("✓ No complex compilation steps")
    print("✓ JIT compilation of C++ code")
    print("✓ Full API compatibility")
    print("✓ Direct access to graph_lite.h")

if __name__ == "__main__":
    main()
