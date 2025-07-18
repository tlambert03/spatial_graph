# Migration from Witty/Cython to cppyy

## Current Architecture (Witty/Cython)

### Process Flow:
1. **Template Generation**: Cheetah templates generate `.pyx` files based on runtime parameters
2. **JIT Compilation**: Witty compiles the generated Cython code to C extension modules  
3. **Template Instantiation**: Each unique combination of types creates a new compiled module
4. **Caching**: Witty caches compiled modules based on source hash

### Key Components:
- `wrapper_template.pyx`: 658-line Cheetah template that generates Cython bindings
- `graph_lite.h`: C++ template library (1358 lines)
- `dtypes.py`: Type system for mapping Python types to C++/Cython types
- Complex Cython memory management and type conversion code

### Pros:
- ✅ Very fast execution (compiled C++)
- ✅ Mature compilation pipeline
- ✅ Good numpy integration through Cython

### Cons:
- ❌ Complex template system with 3 layers (Cheetah → Cython → C++)
- ❌ Large amount of boilerplate code (658 lines of template)
- ❌ Compilation overhead on first use
- ❌ Difficult to debug generated code
- ❌ Template syntax is hard to maintain

## Proposed Architecture (cppyy)

### Process Flow:
1. **Direct Template Instantiation**: cppyy directly instantiates C++ templates at runtime
2. **Dynamic Struct Generation**: Generate C++ structs programmatically as strings
3. **Just-in-Time Binding**: cppyy creates Python bindings on-the-fly
4. **Template Caching**: Cache instantiated C++ template classes

### Key Benefits:

#### 1. **Simplified Architecture**
```python
# Before: 3-layer template system
Cheetah Template → Cython Code → C++ Bindings

# After: Direct C++ template instantiation  
Python Parameters → C++ Template → Python Bindings
```

#### 2. **Reduced Code Complexity**
- **Before**: 658 lines of complex Cheetah/Cython template
- **After**: ~200 lines of straightforward Python code

#### 3. **Better Developer Experience**
```python
# Before: Debug generated Cython code
# - Template syntax errors are hard to trace
# - Generated .pyx files are not human-readable
# - Compilation errors reference generated code

# After: Debug normal Python/C++ code
# - Direct access to C++ objects
# - Clear stack traces
# - Standard C++ debugging tools work
```

#### 4. **More Flexible Type System**
```python
# Before: Fixed template with Cheetah conditionals
%if $directed
    pair[NeighborsIterator, NeighborsIterator] out_neighbors(NodeType& node)
%else  
    pair[NeighborsIterator, NeighborsIterator] neighbors(NodeType& node)
%end if

# After: Dynamic C++ code generation
if directed:
    api_methods.append("auto out_neighbors(NodeType node) -> std::pair<...>")
else:
    api_methods.append("auto neighbors(NodeType node) -> std::pair<...>")
```

#### 5. **Faster Iteration**
```python
# Before: Any template change requires
# 1. Modify .pyx template
# 2. Regenerate Cython code  
# 3. Recompile C extension
# 4. Test changes

# After: Template changes are just Python code
# 1. Modify Python template generator
# 2. Test changes immediately
```

## Migration Strategy

### Phase 1: Parallel Implementation
- Create `graph_cppyy.py` alongside existing `graph.py`
- Implement core functionality (add_node, add_edge, basic queries)
- Create comprehensive test suite comparing both implementations

### Phase 2: Feature Parity
- Implement all current Graph functionality in cppyy version
- Ensure performance is comparable or better
- Handle edge cases and error conditions

### Phase 3: Integration
- Update `SpatialGraph` to use cppyy backend
- Deprecate Witty/Cython implementation
- Update documentation and examples

## Performance Considerations

### Expected Performance Impact:
- **Startup**: Faster (no compilation step)
- **Runtime**: Similar or slightly better (direct C++ calls)
- **Memory**: Similar (same underlying C++ data structures)
- **Template Instantiation**: Much faster (no file I/O or compilation)

### Benchmarking Plan:
```python
def benchmark_comparison():
    # Test both implementations with:
    # - Various graph sizes (1K, 10K, 100K, 1M nodes)
    # - Different attribute configurations
    # - Common operations (add_node, query_neighbors, etc.)
    # - Memory usage patterns
```

## Risk Mitigation

### Potential Issues:
1. **cppyy Learning Curve**: Team needs to learn cppyy patterns
2. **Template Debugging**: Different debugging workflow
3. **Dependencies**: Adding cppyy as a dependency

### Mitigation Strategies:
1. **Gradual Migration**: Keep both implementations during transition
2. **Comprehensive Testing**: Extensive test suite ensuring identical behavior
3. **Documentation**: Clear migration guide and cppyy best practices
4. **Fallback Plan**: Keep Witty implementation as backup option

## Code Example Comparison

### Current (Witty/Cython):
```python
# Complex template generation
wrapper_template = Template(file="wrapper_template.pyx")
wrapper_template.node_dtype = node_dtype
wrapper = witty.compile_module(str(wrapper_template), ...)

# Usage
graph = Graph("uint64", {"position": "double[3]"}, {"weight": "float32"})
```

### Proposed (cppyy):
```python
# Direct template instantiation
graph_class = cppyy.gbl.Graph[uint64_t, NodeData, EdgeData]
cpp_graph = graph_class()

# Usage (same API)
graph = GraphCppyy("uint64", {"position": "double[3]"}, {"weight": "float32"})
```

The cppyy approach offers significant advantages in terms of maintainability, debuggability, and development velocity while preserving the performance characteristics of the current system.
