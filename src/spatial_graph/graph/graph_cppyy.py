"""
New cppyy-based Graph implementation to replace the Witty/Cython approach.
"""
import cppyy
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Any
from spatial_graph.dtypes import DType


class Graph:
    """
    Graph implementation using cppyy for C++ template instantiation.
    
    This replaces the Witty/Cython/Cheetah template approach with direct
    C++ template instantiation via cppyy.
    """
    
    _template_cache = {}  # Cache instantiated templates
    
    def __new__(
        cls,
        node_dtype,
        node_attr_dtypes=None,
        edge_attr_dtypes=None,
        directed=False,
        *args,
        **kwargs,
    ):
        if node_attr_dtypes is None:
            node_attr_dtypes = {}
        if edge_attr_dtypes is None:
            edge_attr_dtypes = {}

        # Convert to DType objects
        node_dtype = DType(node_dtype)
        node_attr_dtypes = {
            name: DType(dtype) for name, dtype in node_attr_dtypes.items()
        }
        edge_attr_dtypes = {
            name: DType(dtype) for name, dtype in edge_attr_dtypes.items()
        }

        # Create a unique key for this template configuration
        config_key = cls._make_config_key(node_dtype, node_attr_dtypes, edge_attr_dtypes, directed)
        
        # Check if we already have this template instantiated
        if config_key not in cls._template_cache:
            cls._setup_cppyy_environment()
            graph_class = cls._instantiate_template(node_dtype, node_attr_dtypes, edge_attr_dtypes, directed)
            cls._template_cache[config_key] = graph_class
        
        # Create instance of the templated class
        graph_class = cls._template_cache[config_key]
        instance = object.__new__(cls)
        instance._cpp_graph = graph_class()
        instance._node_dtype = node_dtype
        instance._node_attr_dtypes = node_attr_dtypes
        instance._edge_attr_dtypes = edge_attr_dtypes
        instance._directed = directed
        
        return instance

    @classmethod
    def _setup_cppyy_environment(cls):
        """Setup cppyy environment - load headers, set compiler flags, etc."""
        if hasattr(cls, '_cppyy_setup_done'):
            return
            
        # Add include directory
        src_dir = Path(__file__).parent / "src"
        cppyy.add_include_path(str(src_dir))
        
        # Set C++20 standard
        cppyy.cppdef("#pragma cling add_cxx_flag \"-std=c++20\"")
        cppyy.cppdef("#pragma cling add_cxx_flag \"-O3\"")
        cppyy.cppdef("#pragma cling add_cxx_flag \"-Wno-unused-variable\"")
        cppyy.cppdef("#pragma cling add_cxx_flag \"-Wno-unreachable-code\"")
        
        # Include the main header
        cppyy.include("graph_lite.h")
        
        cls._cppyy_setup_done = True

    @classmethod
    def _make_config_key(cls, node_dtype, node_attr_dtypes, edge_attr_dtypes, directed):
        """Create a unique key for this template configuration."""
        node_attrs_key = tuple(sorted((k, v.as_string) for k, v in node_attr_dtypes.items()))
        edge_attrs_key = tuple(sorted((k, v.as_string) for k, v in edge_attr_dtypes.items()))
        return (node_dtype.as_string, node_attrs_key, edge_attrs_key, directed)

    @classmethod
    def _instantiate_template(cls, node_dtype, node_attr_dtypes, edge_attr_dtypes, directed):
        """Instantiate the C++ template for the given configuration."""
        
        # Generate NodeData and EdgeData structs
        node_data_struct = cls._generate_data_struct("NodeData", node_attr_dtypes)
        edge_data_struct = cls._generate_data_struct("EdgeData", edge_attr_dtypes)
        
        # Define the structs in cppyy
        cppyy.cppdef(node_data_struct)
        cppyy.cppdef(edge_data_struct)
        
        # Generate the template instantiation
        node_type = node_dtype.base_c_type
        direction = "graph_lite::EdgeDirection::DIRECTED" if directed else "graph_lite::EdgeDirection::UNDIRECTED"
        
        template_instantiation = f"""
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
        
        cppyy.cppdef(template_instantiation)
        
        # Return the instantiated class
        return cppyy.gbl.GraphInstance

    @classmethod
    def _generate_data_struct(cls, struct_name: str, attr_dtypes: Dict[str, DType]) -> str:
        """Generate C++ struct definition for node or edge data."""
        if not attr_dtypes:
            return f"""
            struct {struct_name} {{
                {struct_name}() = default;
            }};
            """
        
        # Generate member declarations
        members = []
        constructor_params = []
        constructor_initializers = []
        
        for name, dtype in attr_dtypes.items():
            members.append(f"    {dtype.to_c_decl(name)};")
            constructor_params.append(f"{dtype.base_c_type} _{name}")
            if dtype.is_array:
                # For arrays, we need to copy element by element
                init_code = "{ " + ", ".join(f"_{name}[{i}]" for i in range(dtype.size)) + " }"
                constructor_initializers.append(f"{name}{init_code}")
            else:
                constructor_initializers.append(f"{name}(_{name})")
        
        constructor_signature = f"{struct_name}(" + ", ".join(constructor_params) + ")"
        constructor_init_list = " : " + ", ".join(constructor_initializers) if constructor_initializers else ""
        
        struct_def = f"""
        struct {struct_name} {{
            {struct_name}() = default;
            {constructor_signature}{constructor_init_list} {{}}
            
{chr(10).join(members)}
        }};
        """
        
        return struct_def

    def __init__(self, node_dtype, node_attr_dtypes=None, edge_attr_dtypes=None, directed=False):
        """Initialize the Graph instance."""
        # Most of the work is done in __new__, just store references
        self.node_dtype = node_dtype
        self.node_attr_dtypes = node_attr_dtypes or {}
        self.edge_attr_dtypes = edge_attr_dtypes or {}
        self.directed = directed
        
        # Create Python wrapper objects for attribute access
        self.node_attrs = NodeAttrs(self)
        self.edge_attrs = EdgeAttrs(self)

    # Core graph operations - these would delegate to self._cpp_graph
    def add_node(self, node_id, **attrs):
        """Add a single node with attributes."""
        node_data = self._create_node_data(attrs)
        return self._cpp_graph.add_node_with_prop(node_id, node_data)
    
    def add_nodes(self, node_ids, **attrs):
        """Add multiple nodes with attributes."""
        for i, node_id in enumerate(node_ids):
            node_attrs = {name: values[i] for name, values in attrs.items()}
            self.add_node(node_id, **node_attrs)
    
    def add_edge(self, source, target, **attrs):
        """Add a single edge with attributes."""
        edge_data = self._create_edge_data(attrs)
        return self._cpp_graph.add_edge_with_prop(source, target, edge_data)
    
    def nodes(self):
        """Get all node IDs as numpy array."""
        # This would need to iterate through the C++ graph and collect node IDs
        node_count = self._cpp_graph.size()
        node_ids = np.empty(node_count, dtype=self._node_dtype.base)
        
        # Iterate through C++ graph (pseudo-code, actual implementation depends on graph_lite API)
        it = self._cpp_graph.begin()
        for i in range(node_count):
            node_ids[i] = it.__deref__()  # Get the node ID
            it.__preinc__()  # Move to next
        
        return node_ids
    
    def _create_node_data(self, attrs):
        """Create a NodeData struct from Python attributes."""
        # Convert Python values to C++ struct
        # This is where you'd handle the dtype conversions
        if not self._node_attr_dtypes:
            return cppyy.gbl.NodeData()
        
        # Convert attrs to match the expected C++ types
        cpp_args = []
        for name, dtype in self._node_attr_dtypes.items():
            value = attrs.get(name)
            if dtype.is_array:
                # Handle array types
                cpp_args.append(value)  # cppyy should handle numpy array conversion
            else:
                cpp_args.append(value)
        
        return cppyy.gbl.NodeData(*cpp_args)
    
    def _create_edge_data(self, attrs):
        """Create an EdgeData struct from Python attributes."""
        if not self._edge_attr_dtypes:
            return cppyy.gbl.EdgeData()
        
        cpp_args = []
        for name, dtype in self._edge_attr_dtypes.items():
            value = attrs.get(name)
            cpp_args.append(value)
        
        return cppyy.gbl.EdgeData(*cpp_args)


# You'd still need the NodeAttrs and EdgeAttrs classes for the Python API
class NodeAttrs:
    def __init__(self, graph):
        self.graph = graph
    
    def __getitem__(self, nodes):
        return NodeAttrsView(self.graph, nodes)


class EdgeAttrs:
    def __init__(self, graph):
        self.graph = graph
    
    def __getitem__(self, edges):
        return EdgeAttrsView(self.graph, edges)


class NodeAttrsView:
    def __init__(self, graph, nodes):
        self.graph = graph
        self.nodes = nodes
    
    def __getattr__(self, name):
        if name in self.graph._node_attr_dtypes:
            # Get attribute values from C++ graph
            return self._get_node_attr(name)
        raise AttributeError(name)
    
    def _get_node_attr(self, name):
        # Implementation would call into C++ graph to get node attributes
        pass


class EdgeAttrsView:
    def __init__(self, graph, edges):
        self.graph = graph
        self.edges = edges
    
    def __getattr__(self, name):
        if name in self.graph._edge_attr_dtypes:
            return self._get_edge_attr(name)
        raise AttributeError(name)
    
    def _get_edge_attr(self, name):
        # Implementation would call into C++ graph to get edge attributes
        pass
