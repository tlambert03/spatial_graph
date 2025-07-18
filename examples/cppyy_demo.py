"""
Example demonstrating the cppyy approach for spatial_graph.

This shows how the same API could be achieved with much simpler code.
"""
import cppyy
import numpy as np

# Setup cppyy environment (this would be done once)
def setup_cppyy():
    """Initialize cppyy with spatial_graph headers."""
    cppyy.cppdef("#pragma cling add_cxx_flag \"-std=c++20\"")
    cppyy.cppdef("#pragma cling add_cxx_flag \"-O3\"")
    
    # Include the graph_lite header
    # In real implementation, this would use the actual path
    cppyy.cppdef("""
    #include <vector>
    #include <unordered_map>
    
    // Simplified version of graph_lite for demonstration
    namespace graph_lite {
        enum class EdgeDirection { DIRECTED, UNDIRECTED };
        enum class MultiEdge { ALLOWED, DISALLOWED };
        enum class SelfLoop { ALLOWED, DISALLOWED };
        enum class Map { UNORDERED_MAP };
        enum class Container { VEC };
        
        template<typename NodeType, typename NodeData, typename EdgeData,
                 EdgeDirection direction, MultiEdge multi_edge, SelfLoop self_loop,
                 Map map_type, Container container>
        class Graph {
        private:
            std::unordered_map<NodeType, NodeData> nodes_;
            std::vector<std::tuple<NodeType, NodeType, EdgeData>> edges_;
            
        public:
            int add_node_with_prop(const NodeType& node, const NodeData& data) {
                nodes_[node] = data;
                return 0;
            }
            
            int add_edge_with_prop(const NodeType& u, const NodeType& v, const EdgeData& data) {
                edges_.emplace_back(u, v, data);
                return 0;
            }
            
            NodeData& node_prop(const NodeType& node) {
                return nodes_[node];
            }
            
            EdgeData& edge_prop(const NodeType& u, const NodeType& v) {
                // Simplified - would need proper edge lookup
                for (auto& edge : edges_) {
                    if (std::get<0>(edge) == u && std::get<1>(edge) == v) {
                        return std::get<2>(edge);
                    }
                }
                throw std::runtime_error("Edge not found");
            }
            
            size_t size() const { return nodes_.size(); }
            size_t num_edges() const { return edges_.size(); }
        };
    }
    """)

def create_graph_type(node_dtype_str, node_attrs, edge_attrs, directed=False):
    """
    Create a specialized graph type using cppyy template instantiation.
    
    This replaces the entire Cheetah template + Cython compilation process.
    """
    
    # Map Python type strings to C++ types
    type_mapping = {
        "uint64": "uint64_t",
        "int32": "int32_t", 
        "float32": "float",
        "double": "double"
    }
    
    node_type = type_mapping[node_dtype_str]
    
    # Generate NodeData struct
    if node_attrs:
        node_fields = []
        node_constructor_params = []
        node_constructor_init = []
        
        for name, dtype_str in node_attrs.items():
            if "[" in dtype_str:
                # Array type like "double[3]"
                base_type, size = dtype_str.split("[")
                size = int(size.rstrip("]"))
                cpp_type = type_mapping[base_type]
                node_fields.append(f"{cpp_type} {name}[{size}];")
                node_constructor_params.append(f"const {cpp_type}* _{name}")
                init_code = f"std::copy(_{name}, _{name} + {size}, {name})"
                node_constructor_init.append(init_code)
            else:
                # Scalar type
                cpp_type = type_mapping[dtype_str]
                node_fields.append(f"{cpp_type} {name};")
                node_constructor_params.append(f"{cpp_type} _{name}")
                node_constructor_init.append(f"{name}(_{name})")
        
        node_struct = f"""
        struct NodeData {{
            NodeData() = default;
            NodeData({', '.join(node_constructor_params)}) {{
                {'; '.join(node_constructor_init)};
            }}
            {' '.join(node_fields)}
        }};
        """
    else:
        node_struct = "struct NodeData { NodeData() = default; };"
    
    # Generate EdgeData struct  
    if edge_attrs:
        edge_fields = []
        edge_constructor_params = []
        edge_constructor_init = []
        
        for name, dtype_str in edge_attrs.items():
            cpp_type = type_mapping[dtype_str]
            edge_fields.append(f"{cpp_type} {name};")
            edge_constructor_params.append(f"{cpp_type} _{name}")
            edge_constructor_init.append(f"{name}(_{name})")
        
        edge_struct = f"""
        struct EdgeData {{
            EdgeData() = default;
            EdgeData({', '.join(edge_constructor_params)}) : {', '.join(edge_constructor_init)} {{}}
            {' '.join(edge_fields)}
        }};
        """
    else:
        edge_struct = "struct EdgeData { EdgeData() = default; };"
    
    # Define the structs in cppyy
    cppyy.cppdef(node_struct)
    cppyy.cppdef(edge_struct)
    
    # Instantiate the template
    direction = "graph_lite::EdgeDirection::DIRECTED" if directed else "graph_lite::EdgeDirection::UNDIRECTED"
    
    template_def = f"""
    using GraphInstance = graph_lite::Graph<
        {node_type},
        NodeData, 
        EdgeData,
        {direction},
        graph_lite::MultiEdge::DISALLOWED,
        graph_lite::SelfLoop::DISALLOWED,
        graph_lite::Map::UNORDERED_MAP,
        graph_lite::Container::VEC
    >;
    """
    
    cppyy.cppdef(template_def)
    
    return cppyy.gbl.GraphInstance


class GraphCppyySimple:
    """Simplified Graph class using cppyy - demonstrates the concept."""
    
    def __init__(self, node_dtype, node_attr_dtypes=None, edge_attr_dtypes=None, directed=False):
        self.node_dtype = node_dtype
        self.node_attr_dtypes = node_attr_dtypes or {}
        self.edge_attr_dtypes = edge_attr_dtypes or {}
        self.directed = directed
        
        # Create the C++ graph instance
        graph_class = create_graph_type(node_dtype, node_attr_dtypes, edge_attr_dtypes, directed)
        self._cpp_graph = graph_class()
    
    def add_node(self, node_id, **attrs):
        """Add a node with attributes."""
        if self.node_attr_dtypes:
            # Create NodeData with attributes
            attr_values = []
            for name, dtype in self.node_attr_dtypes.items():
                value = attrs[name]
                if "[" in dtype:  # Array type
                    attr_values.append(value.ctypes.data)  # Pass pointer to numpy array
                else:
                    attr_values.append(value)
            node_data = cppyy.gbl.NodeData(*attr_values)
        else:
            node_data = cppyy.gbl.NodeData()
        
        return self._cpp_graph.add_node_with_prop(node_id, node_data)
    
    def add_edge(self, source, target, **attrs):
        """Add an edge with attributes."""
        if self.edge_attr_dtypes:
            attr_values = [attrs[name] for name in self.edge_attr_dtypes.keys()]
            edge_data = cppyy.gbl.EdgeData(*attr_values)
        else:
            edge_data = cppyy.gbl.EdgeData()
        
        return self._cpp_graph.add_edge_with_prop(source, target, edge_data)
    
    def size(self):
        """Get number of nodes."""
        return self._cpp_graph.size()
    
    def num_edges(self):
        """Get number of edges."""
        return self._cpp_graph.num_edges()


def demo():
    """Demonstrate the cppyy approach."""
    print("Setting up cppyy environment...")
    setup_cppyy()
    
    print("Creating graph with node positions and edge weights...")
    graph = GraphCppyySimple(
        node_dtype="uint64",
        node_attr_dtypes={"position": "double[3]"},
        edge_attr_dtypes={"weight": "float32"},
        directed=False
    )
    
    print("Adding nodes...")
    positions = [
        np.array([1.0, 2.0, 3.0], dtype=np.float64),
        np.array([4.0, 5.0, 6.0], dtype=np.float64),
        np.array([7.0, 8.0, 9.0], dtype=np.float64)
    ]
    
    for i, pos in enumerate(positions):
        graph.add_node(i, position=pos)
    
    print("Adding edges...")
    graph.add_edge(0, 1, weight=1.5)
    graph.add_edge(1, 2, weight=2.5)
    
    print(f"Graph has {graph.size()} nodes and {graph.num_edges()} edges")
    print("✅ Success! The cppyy approach works.")


if __name__ == "__main__":
    demo()
