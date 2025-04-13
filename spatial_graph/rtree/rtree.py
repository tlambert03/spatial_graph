from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Generic, Literal, TypeVar, overload
import sys
import witty
import numpy as np
import numpy.typing as npt
from Cheetah.Template import Template
from pathlib import Path
from ..dtypes import DType

if TYPE_CHECKING:
    from typing_extensions import Self

DEFINE_MACROS = [("RTREE_NOATOMICS", "1")] if sys.platform == "win32" else []

CT = TypeVar("CT", bound=np.number)
IT = TypeVar("IT", bound=np.number)


class RTree(Generic[CT, IT]):
    """A generic RTree implementation, compiled on-the-fly during
    instantiation.

    Args:

        item_dtype (``string``):

            The C type of the items to hold. Can be a scalar (e.g. ``uint64``)
            or an array of scalars (e.g., "uint64[3]").

        coord_dtype (``string``):

            The scalar C type to use for coordinates (e.g., ``float``).

        dims (``int``):

            The dimension of the r-tree.

    Subclassing:

        This generic implementation can be subclassed and modified in the
        following ways:

        The class members ``pyx_item_t_declaration`` and
        ``c_item_t_declaration`` can be overwritten to use custom ``item_t``
        structures. This will also require overwriting the
        ``c_converter_functions`` to translate between the PYX interface (where
        items are scalars or C arrays of scalars) and the C interface (the
        custom ``item_t`` type.

        The following constants and typedefs are available to use in the
        provided code:

            DIMS:

                A constant set to the value of ``dims``.

            item_base_t:

                The scalar type of the item (e.g., ``uint64``), regardless of
                whether this is a scalar or array item.
    """

    # overwrite in subclasses for custom item_t structures
    pyx_item_t_declaration: ClassVar[str] = ""
    c_item_t_declaration: ClassVar[str] = ""

    # overwrite in subclasses for custom converters
    c_converter_functions: ClassVar[str] = ""

    # overwrite in subclasses for custom item comparison code
    c_equal_function: ClassVar[str] = ""

    # overwrite in subclasses for custom distance computation
    c_distance_function: ClassVar[str] = ""

    coord_dtype: DType
    item_dtype: DType

    def __new__(
        cls, item_dtype: npt.DTypeLike, coord_dtype: npt.DTypeLike, dims: int
    ) -> Self:
        ############################################
        # create wrapper from template and compile #
        ############################################
        _coord_dtype = DType(coord_dtype)
        _item_dtype = DType(item_dtype)

        src_dir = Path(__file__).parent
        wrapper_template = Template(
            file=str(src_dir / "wrapper_template.pyx"),
            compilerSettings={"directiveStartToken": "%"},
        )
        wrapper_template.item_dtype = _item_dtype
        wrapper_template.coord_dtype = _coord_dtype
        wrapper_template.dims = dims
        wrapper_template.c_distance_function = cls.c_distance_function
        wrapper_template.pyx_item_t_declaration = cls.pyx_item_t_declaration
        wrapper_template.c_item_t_declaration = cls.c_item_t_declaration
        wrapper_template.c_converter_functions = cls.c_converter_functions
        wrapper_template.c_equal_function = cls.c_equal_function

        wrapper = witty.compile_module(
            str(wrapper_template),
            source_files=[
                src_dir / "src" / "rtree.h",
                src_dir / "src" / "rtree.c",
                src_dir / "src" / "config.h",
            ],
            extra_compile_args=["/O2" if sys.platform == "win32" else "-O3"],
            include_dirs=[str(src_dir)],
            language="c",
            quiet=True,
            define_macros=DEFINE_MACROS,
            output_dir=str(Path(__file__).parent),
            name=f"rtree_{_item_dtype.base}_{_coord_dtype.base}_{dims}d".lower(),
        )
        RTreeType = type(cls.__name__, (cls, wrapper.RTree), {})
        obj = wrapper.RTree.__new__(RTreeType)
        obj.coord_dtype = _coord_dtype
        obj.item_dtype = _item_dtype
        return obj

    def insert_point_item(self, item: IT, position: npt.NDArray[CT]) -> None:
        items = np.array([item], dtype=self.item_dtype.base)
        positions = position[np.newaxis]
        return self.insert_point_items(items, positions)

    def delete_item(self, item, bb_min, bb_max=None):
        items = np.array([item], dtype=self.item_dtype.base)
        bb_mins = bb_min[np.newaxis, :]
        bb_maxs = None if bb_max is None else bb_max[np.newaxis, :]
        return self.delete_items(items, bb_mins, bb_maxs)

    if TYPE_CHECKING:

        def insert_point_items(
            self, items: npt.NDArray[IT], points: npt.NDArray[CT]
        ) -> None: ...
        def insert_bb_items(
            self,
            items: npt.NDArray[IT],
            bb_mins: npt.NDArray[CT],
            bb_maxs: npt.NDArray[CT],
        ) -> None: ...
        def count(self, bb_min: npt.NDArray[CT], bb_max: npt.NDArray[CT]) -> int: ...
        def bounding_box(self) -> tuple[npt.NDArray[CT], npt.NDArray[CT]]: ...
        def search(
            self, bb_min: npt.NDArray[CT], bb_max: npt.NDArray[CT]
        ) -> npt.NDArray[IT]: ...
        @overload
        def nearest(
            self, point: npt.NDArray[CT], k: int, return_distances: Literal[True]
        ) -> tuple[npt.NDArray[IT], npt.NDArray[CT]]: ...
        @overload
        def nearest(
            self, point: npt.NDArray[CT], k: int, return_distances: Literal[False] = ...
        ) -> npt.NDArray[IT]: ...
        def nearest(
            self, point: npt.NDArray[CT], k: int, return_distances: bool = False
        ) -> npt.NDArray[IT] | tuple[npt.NDArray[IT], npt.NDArray[CT]]: ...
        def delete_items(
            self,
            items: npt.NDArray[IT],
            bb_mins: npt.NDArray[CT],
            bb_maxs: npt.NDArray[CT] | None = None,
        ) -> int: ...
        def __len__(self) -> int: ...


class RTree2D(RTree[CT, IT]):
    def __new__(cls, item_dtype: npt.DTypeLike, coord_dtype: npt.DTypeLike) -> Self:
        return super().__new__(cls, item_dtype, coord_dtype, 2)


class RTree2DInt(RTree2D[np.int64, IT], Generic[IT]):
    def __new__(cls, item_dtype: str) -> Self:
        return super().__new__(cls, item_dtype, np.int64)


class RTree2DFloat(RTree2D[np.float64, IT], Generic[IT]):
    def __new__(cls, item_dtype: str) -> Self:
        return super().__new__(cls, item_dtype, np.float64)


class RTree2DFloat32(RTree2D[np.float32, IT], Generic[IT]):
    def __new__(cls, item_dtype: str) -> Self:
        return super().__new__(cls, item_dtype, np.float32)


class RTree3D(RTree[CT, IT]):
    def __new__(cls, item_dtype: npt.DTypeLike, coord_dtype: npt.DTypeLike) -> Self:
        return super().__new__(cls, item_dtype, coord_dtype, 3)


class RTree3DInt(RTree3D[np.int64, IT], Generic[IT]):
    def __new__(cls, item_dtype: str) -> Self:
        return super().__new__(cls, item_dtype, np.int64)


class RTree3DFloat(RTree3D[np.float64, IT], Generic[IT]):
    def __new__(cls, item_dtype: str) -> Self:
        return super().__new__(cls, item_dtype, np.float64)


class RTree3DFloat32(RTree3D[np.float32, IT], Generic[IT]):
    def __new__(cls, item_dtype: str) -> Self:
        return super().__new__(cls, item_dtype, np.float32)


class RTree4D(RTree[CT, IT]):
    def __new__(cls, item_dtype: npt.DTypeLike, coord_dtype: npt.DTypeLike) -> Self:
        return super().__new__(cls, item_dtype, coord_dtype, 4)


class RTree4DInt(RTree4D[np.int64, IT], Generic[IT]):
    def __new__(cls, item_dtype: str) -> Self:
        return super().__new__(cls, item_dtype, np.int64)


class RTree4DFloat(RTree4D[np.float64, IT], Generic[IT]):
    def __new__(cls, item_dtype: str) -> Self:
        return super().__new__(cls, item_dtype, np.float64)


class RTree4DFloat32(RTree4D[np.float32, IT], Generic[IT]):
    def __new__(cls, item_dtype: str) -> Self:
        return super().__new__(cls, item_dtype, np.float32)
