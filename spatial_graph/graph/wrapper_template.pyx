from cython cimport view
from cython.operator cimport dereference as deref, preincrement as inc
from libc.stdint cimport *
from libcpp.utility cimport pair
import numpy as np


cdef extern from *:
    """
    #include "src/graph_lite.h"

    // partial template instantiation of graph_lite::Graph as
    // UndirectedGraphTmpl
    template<typename NodeType, typename NodeData, typename EdgeData>
    class UndirectedGraphTmpl : public graph_lite::Graph<
        NodeType,
        NodeData,
        EdgeData,
        graph_lite::EdgeDirection::UNDIRECTED,
        graph_lite::MultiEdge::DISALLOWED,
        graph_lite::SelfLoop::DISALLOWED,
        graph_lite::Map::UNORDERED_MAP,
        graph_lite::Container::VEC
    > {};
    """

    cdef cppclass UndirectedGraphTmpl[NodeType, NodeData, EdgeData]:

        cppclass Iterator:
            NodeType operator*()
            Iterator operator++()
            bint operator==(Iterator)
            bint operator!=(Iterator)

        cppclass EdgePropIterWrap[ED]:
            ED& prop()

        cppclass NeighborsIterator:
            pair[NodeType, EdgePropIterWrap[EdgeData]] operator*()
            NeighborsIterator operator++()
            bint operator==(NeighborsIterator)
            bint operator!=(NeighborsIterator)

        int add_node_with_prop(NodeType& node, NodeData& prop)

        int add_edge_with_prop(NodeType& source, NodeType& target, EdgeData& prop)

        NodeData& node_prop[T](T& node)

        EdgeData& edge_prop[T](T& u, T& v)

        pair[NeighborsIterator, NeighborsIterator] neighbors(Iterator& node)
        pair[NeighborsIterator, NeighborsIterator] neighbors(NodeType& node)

        int remove_nodes(NodeType& node)

        int count_neighbors(NodeType& node)

        size_t size() const

        size_t num_edges() const

        Iterator begin()

        Iterator end()

ctypedef $node_dtype.to_pyxtype() NodeType

%for class_name, dtypes in [
    ("NodeData", $node_attr_dtypes),
    ("EdgeData", $edge_attr_dtypes)
]
cdef extern from *:
    """
    class $class_name {
        public:
            ${class_name}() {};
            ${class_name}(
                %set $sep=""
                %for name, dtype in $dtypes.items()
                $sep$dtype.to_c_decl("_" + $name)
                %set $sep=", "
                %end for
            ) :
            %set $sep=""
            %for name, dtype in $dtypes.items()
            $sep
            %if $dtype.is_array
            ${name}{
            %set $isep=""
            %for i in range($dtype.size)
                ${isep}_${name}[$i]
                %set $isep=", "
            %end for
            }
            %else
            ${name}(_$name)%slurp
            %end if
            %set $sep=", "
            %end for
            {};

            %for name, dtype in $dtypes.items()
            $dtype.to_c_decl($name);
            %end for
    };
    """

    cdef cppclass $class_name:

        ${class_name}(
            %set $sep=""
            %for name, dtype in $dtypes.items()
            $sep$dtype.to_c_decl("_" + $name)
            %set $sep=", "
            %end for
        ) except +

        %for name, dtype in $dtypes.items()
        $dtype.to_pyxtype() $name
        %end for

cdef class ${class_name}View:

    cdef $class_name* _ptr

    cdef set_ptr(self, $class_name* ptr):
        self._ptr = ptr

    %for name, dtype in $dtypes.items():
    @property
    def ${name}(self):
        %if $dtype.is_array
        return <${dtype.base_c_type}[:${dtype.size}]>(self._ptr.${name})
        %else
        return self._ptr.$name
        %end if

    @${name}.setter
    def ${name}(self, value):
        self._ptr.$name = value

    %end for

%end for

ctypedef UndirectedGraphTmpl[NodeType, NodeData, EdgeData] UndirectedGraphType
ctypedef UndirectedGraphType.Iterator NodeIterator
ctypedef UndirectedGraphType.NeighborsIterator NeighborsIterator


cdef class UndirectedGraph:

    cdef UndirectedGraphType _graph

    %for kind, Kind, dtypes in [
        ("node", "Node", $node_attr_dtypes),
        ("edge", "Edge", $edge_attr_dtypes)
    ]
    def add_${kind}(
            self,
            %if kind == "node"
            NodeType node,
            %else
            NodeType[:] edge,
            %end if
            %set sep=""
            %for name, dtype in $dtypes.items()
            $sep${dtype.to_pyxtype(use_memory_view=True)} $name
            %set $sep=", "
            %end for
    ):

        %for name, dtype in $dtypes.items()
        %if dtype.is_array
        cdef ${dtype.to_pyxtype()} _p_${name} = &${name}[0]
        %end if
        %end for

        self._graph.add_${kind}_with_prop(
            %if kind == "node"
            node,
            %else
            edge[0], edge[1],
            %end if
            ${Kind}Data(
                %set sep=""
                %for name, dtype in $dtypes.items()
                %if $dtype.is_array
                ${sep}_p_${name}%slurp
                %else
                ${sep}${name}%slurp
                %set sep=", "
                %end if
                %end for

            )
        )

    def add_${kind}s(
            self,
            %if kind == "node"
            NodeType[::1] nodes,
            %else
            NodeType[:, :] edges,
            %end if
            %set sep=""
            %for name, dtype in $dtypes.items()
            $sep${dtype.to_pyxtype(use_memory_view=True, add_dim=True)} $name
            %set $sep=", "
            %end for
    ):

        %for name, dtype in $dtypes.items()
        %if dtype.is_array
        cdef ${dtype.to_pyxtype()} _p_${name}
        %end if
        %end for

        for i in range(len(${kind}s)):
            %for name, dtype in $dtypes.items()
            %if $dtype.is_array
            _p_${name} = &${name}[i, 0]
            %end if
            %end for
            self._graph.add_${kind}_with_prop(
                %if kind == "node"
                nodes[i],
                %else
                edges[i, 0], edges[i, 1],
                %end if
                ${Kind}Data(
                    %set sep=""
                    %for name, dtype in $dtypes.items()
                    %if $dtype.is_array
                    ${sep}_p_${name}%slurp
                    %else
                    ${sep}${name}[i]%slurp
                    %end if
                    %set $sep=", "
                    %end for

                )
            )
    %end for

    def nodes(self, bint data=False):

        cdef NodeIterator it = self._graph.begin()
        cdef NodeIterator end = self._graph.end()
        cdef NodeDataView node_data = NodeDataView()

        if data:
            while it != end:
                node_data.set_ptr(&self._graph.node_prop(it))
                yield deref(it), node_data
                inc(it)
        else:
            while it != end:
                yield deref(it)
                inc(it)

    def edges(self, node=None, bint data=False):

        if node is not None:
            yield from self._neighbors(<NodeType>node, data)
            return

        # iterate over all edges by iterating over all nodes u and their
        # neighbors v with u < v
        cdef NodeIterator node_it = self._graph.begin()
        cdef NodeIterator node_end = self._graph.end()
        cdef pair[NeighborsIterator, NeighborsIterator] view
        cdef NodeType u, v
        cdef EdgeDataView edge_data = EdgeDataView()

        while node_it != node_end:
            view = self._graph.neighbors(node_it)
            u = deref(node_it)
            it = view.first
            end = view.second
            while it != end:
                v = deref(it).first
                if u < v:
                    if data:
                        edge_data.set_ptr(&deref(it).second.prop())
                        yield (u, v), edge_data
                    else:
                        yield (u, v)
                inc(it)
            inc(node_it)

    # same as above, but for fast access to edges incident to an array of nodes
    def edges_by_nodes(self, NodeType[::1] nodes):

        # iterate over all edges by iterating over all nodes u and their
        # neighbors v with u < v
        cdef pair[NeighborsIterator, NeighborsIterator] view
        cdef NodeType u, v
        cdef Py_ssize_t i = 0

        num_edges = self._num_edges(nodes)
        data = np.empty(shape=(num_edges, 2), dtype="$node_dtype.base")
        cdef NodeType[:, ::1] edges = data

        for u in nodes:
            view = self._graph.neighbors(u)
            it = view.first
            end = view.second
            while it != end:
                v = deref(it).first
                if u < v:
                    edges[i, 0] = u
                    edges[i, 1] = v
                    i += 1
                inc(it)

        return data[:i]

    # generator access to node and edge data

    def nodes_data(self, NodeType[::1] nodes):
        cdef NodeDataView node_data = NodeDataView()
        for node in nodes:
            node_data.set_ptr(&self._graph.node_prop(node))
            yield node, node_data

    def edges_data(self, NodeType[::1] us, NodeType[::1] vs):
        cdef EdgeDataView edge_data = EdgeDataView()
        num_edges = len(us)
        for i in range(num_edges):
            edge_data.set_ptr(&self._graph.edge_prop(us[i], vs[i]))
            yield edge_data

    # access to individual attributes (single and multiple nodes/edges)

    %for name, dtype in $node_attr_dtypes.items()
    def get_node_data_${name}(self, NodeType node):
        %if $dtype.is_array
        return np.array(self._graph.node_prop(node).${name})
        %else
        return self._graph.node_prop(node).${name}
        %end if

    def get_nodes_data_${name}(self, NodeType[:] nodes):

        cdef NodeIterator node_it = self._graph.begin()
        cdef NodeIterator node_end = self._graph.end()
        cdef Py_ssize_t i = 0
        cdef NodeData* node_data

        # allocate array for data
        cdef Py_ssize_t num_nodes = 0
        if nodes is None:
            num_nodes = self._graph.size()
        else:
            num_nodes = len(nodes)
        data = np.empty(shape=(num_nodes,) + $dtype.shape, dtype="$dtype.base")
        cdef $dtype.to_pyxtype(add_dim=True) view = data

        # all nodes requested
        if nodes is None:
            while node_it != node_end:
                %if $dtype.is_array
                node_data = &self._graph.node_prop(node_it)
                %for j in range($dtype.size)
                view[i, $j] = node_data.${name}[$j]
                %end for
                %else
                view[i] = self._graph.node_prop(node_it).${name}
                %end if
                inc(node_it)
                i += 1
        else:
            for i in range(num_nodes):
                %if $dtype.is_array
                node_data = &self._graph.node_prop(nodes[i])
                %for j in range($dtype.size)
                view[i, $j] = node_data.${name}[$j]
                %end for
                %else
                view[i] = self._graph.node_prop(nodes[i]).${name}
                %end if

        return data

    def set_node_data_${name}(
            self,
            NodeType node,
            $dtype.to_pyxtype(use_memory_view=True) $name):
        self._graph.node_prop(node).${name} = $dtype.to_rvalue(name=$name)

    def set_nodes_data_${name}(
            self,
            NodeType[:] nodes,
            $dtype.to_pyxtype(add_dim=True) $name):

        cdef NodeIterator node_it = self._graph.begin()
        cdef NodeIterator node_end = self._graph.end()
        cdef Py_ssize_t i = 0

        # all nodes requested
        %set $rvalue=$dtype.to_rvalue(name=$name, array_index="i")
        if nodes is None:
            while node_it != node_end:
                self._graph.node_prop(node_it).$name = $rvalue
                inc(node_it)
                i += 1
        else:
            assert len(nodes) == len($name)
            for i in range(len(nodes)):
                self._graph.node_prop(nodes[i]).$name = $rvalue
    %end for

    %for name, dtype in $edge_attr_dtypes.items()
    def get_edge_data_${name}(self, NodeType u, NodeType v):
        %if $dtype.is_array
        return np.array(self._graph.edge_prop(u, v).${name})
        %else
        return self._graph.edge_prop(u, v).${name}
        %end if

    def get_edges_data_${name}(self, NodeType[::1] us, NodeType[::1] vs):

        cdef Py_ssize_t i = 0
        cdef Py_ssize_t num_edges = 0
        cdef NodeIterator node_it = self._graph.begin()
        cdef NodeIterator node_end = self._graph.end()
        cdef pair[NeighborsIterator, NeighborsIterator] edges_view
        cdef NodeType u, v
        cdef EdgeData* edge_data

        # allocate array for data
        if us is None and vs is None:
            num_edges = self._graph.num_edges()
        elif us is None or vs is None:
            raise RuntimeError("Either both us and vs are None, or neither")
        else:
            num_edges = len(us)
        data = np.empty(shape=(num_edges,) + $dtype.shape, dtype="$dtype.base")
        cdef $dtype.to_pyxtype(add_dim=True) view = data

        if us is None:

            # iterate over all edges by iterating over all nodes u and their
            # neighbors v with u < v

            while node_it != node_end:
                edges_view = self._graph.neighbors(node_it)
                u = deref(node_it)
                it = edges_view.first
                end = edges_view.second
                while it != end:
                    v = deref(it).first
                    if u < v:
                        %if $dtype.is_array
                        edge_data = &deref(it).second.prop()
                        %for j in range($dtype.size)
                        view[i, $j] = edge_data.${name}[$j]
                        %end for
                        %else
                        view[i] = deref(it).second.prop().$name
                        %end if
                        i += 1
                    inc(it)
                inc(node_it)

        else:
            for i in range(num_edges):
                %if $dtype.is_array
                edge_data = &self._graph.edge_prop(us[i], vs[i])
                %for j in range($dtype.size)
                view[i, $j] = edge_data.${name}[$j]
                %end for
                %else
                view[i] = self._graph.edge_prop(us[i], vs[i]).$name
                %end if

        return data

    def set_edge_data_${name}(
            self,
            NodeType u, NodeType v,
            $dtype.to_pyxtype(use_memory_view=True) $name):
        self._graph.edge_prop(u, v).${name} = $dtype.to_rvalue(name=$name)

    def set_edges_data_${name}(
            self,
            NodeType[::1] us, NodeType[::1] vs,
            $dtype.to_pyxtype(add_dim=True) $name):

        cdef Py_ssize_t i = 0
        cdef Py_ssize_t num_edges = 0

        assert len(us) == len(vs)
        num_edges = len(us)

        %set $rvalue = $dtype.to_rvalue(name=$name, array_index="i")
        for i in range(num_edges):
            self._graph.edge_prop(us[i], vs[i]).$name = $rvalue
    %end for

    # modify graph

    def remove_node(self, NodeType node):
        self._graph.remove_nodes(node)

    # read-only graph properties

    def count_neighbors(self, NodeType[:] nodes):
        num_nodes = len(nodes)
        cdef int[:] counts = view.array(
            shape=(num_nodes,),
            itemsize=sizeof(int),
            format="i")
        for i in range(num_nodes):
            counts[i] = self._graph.count_neighbors(nodes[i])
        return counts

    def _neighbors(self, NodeType node, bint data):

        cdef pair[NeighborsIterator, NeighborsIterator] view = self._graph.neighbors(node)
        cdef NeighborsIterator it = view.first
        cdef NeighborsIterator end = view.second
        cdef EdgeDataView edge_data = EdgeDataView()

        if data:
            while it != end:
                edge_data.set_ptr(&deref(it).second.prop())
                yield deref(it).first, edge_data
                inc(it)
        else:
            while it != end:
                yield deref(it).first
                inc(it)

    def __len__(self):
        return self._graph.size()

    def num_edges(self):
        return self._graph.num_edges()

    def _num_edges(self, NodeType[::1] nodes):
        return np.sum(self.count_neighbors(nodes))