"""Helper functions for DOLFINx."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Callable, Optional, Union

import dolfinx
import matplotlib.pyplot as plt
import numpy as np
import ufl
from dolfinx.fem import Constant, DirichletBCMetaClass, Function, FunctionSpace
from dolfinx.fem.petsc import LinearProblem
from matplotlib.tri import Triangulation

if TYPE_CHECKING:
    from dolfinx.mesh import Mesh
    from matplotlib.colors import Colormap
    from numpy.typing import NDArray


def get_matplotlib_triangulation_from_mesh(mesh: Mesh) -> Triangulation:
    """
    Get Matplotlib triangulation corresponding to DOLFINx mesh.

    Args:
        mesh: Finite element mesh to get triangulation for.

    Returns:
        Object representing triangulation of mesh to use in Matplotlib plot functions.
    """
    if mesh.topology.dim != 2:
        msg = "Only two-dimensional spatial domains are supported"
        raise ValueError(msg)
    # The triangulation of the mesh corresponds to the connectivity between elements of
    # dimension 2 (triangles) and elements of dimension 0 (points)
    mesh.topology.create_connectivity(2, 0)
    triangles = mesh.topology.connectivity(2, 0).array.reshape((-1, 3))
    return Triangulation(
        mesh.geometry.x[:, 0],
        mesh.geometry.x[:, 1],
        triangles,
    )


def project_expression_on_function_space(
    expression: Union[
        ufl.core.expr.Expr,
        Callable[[ufl.SpatialCoordinate], ufl.core.expr.Expr],
    ],
    function_space: ufl.FunctionSpace,
) -> Function:
    """
    Project expression onto finite element function space.

    Args:
        expression: UFL object defining expression to project or function accepting
            single argument corresponding to spatial coordinate vector defining
            expression to project.
        function_space: Finite element function space.

    Returns:
        Function representing projection of expression.
    """
    if not isinstance(expression, ufl.core.expr.Expr):
        mesh = function_space.mesh
        spatial_coordinate = ufl.SpatialCoordinate(mesh)
        expression = expression(spatial_coordinate)
    trial_function = ufl.TrialFunction(function_space)
    test_function = ufl.TestFunction(function_space)
    return LinearProblem(
        ufl.inner(trial_function, test_function) * ufl.dx,
        ufl.inner(expression, test_function) * ufl.dx,
    ).solve()


def evaluate_function_at_points(
    function: Function,
    points: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Evaluate a finite element function at one or more points.

    Args:
        function: Finite element function to evaluate.
        points: One or more points in domain of function to evaluate at. Should be
            either a one-dimensional array corresponding to a single point (with size
            equal to the geometric dimension or 3) or a two-dimensional array
            corresponding to one point per row (with size of last axis equal to the
            geometric dimension or 3).

    Returns:
        Value(s) of function evaluated at point(s).
    """
    mesh = function.function_space.mesh
    if points.ndim not in (1, 2):
        msg = "points argument should be one or two-dimensional array"
        raise ValueError(msg)
    if points.shape[-1] not in (3, mesh.geometry.dim):
        msg = "Last axis of points argument should be of size 3 or spatial dimension"
        raise ValueError(msg)
    if points.ndim == 1:
        points = points[None]
    if points.shape[-1] != 3:
        padded_points = np.zeros(points.shape[:-1] + (3,))
        padded_points[..., : points.shape[-1]] = points
        points = padded_points
    tree = dolfinx.geometry.BoundingBoxTree(mesh, mesh.geometry.dim)
    cell_candidates = dolfinx.geometry.compute_collisions(tree, points)
    if not np.all(cell_candidates.offsets[1:] > 0):
        msg = "One or more points not within domain"
        raise ValueError(msg)
    cell_adjacency_list = dolfinx.geometry.compute_colliding_cells(
        mesh,
        cell_candidates,
        points,
    )
    first_cell_indices = cell_adjacency_list.array[cell_adjacency_list.offsets[:-1]]
    return np.squeeze(function.eval(points, first_cell_indices))


def _preprocess_functions(
    functions: Union[Function, Sequence[Function], dict[str, Function]],
) -> list[tuple[str, Function]]:
    if isinstance(functions, Function):
        return [(functions.name, functions)]
    elif isinstance(functions, dict):
        return list(functions.items())
    else:
        return [(f.name, f) for f in functions]


def plot_1d_functions(
    functions: Union[Function, Sequence[Function], dict[str, Function]],
    *,
    points: Optional[NDArray[np.float64]] = None,
    axis_size: tuple[float, float] = (5.0, 5.0),
    arrangement: Literal["horizontal", "vertical", "stacked"] = "horizontal",
) -> plt.Figure:
    """
    Plot one or more finite element functions on 1D domains using Matplotlib.

    Args:
        functions: A single finite element function, sequence of functions or dictionary
            mapping from string labels to finite element functions, in all cases
            corresponding to the function(s) to plot. If a single function or sequence
            of functions are specified the function :py:attr:`name` attribute(s) will be
            used to set the title for each axis.
        points: Points to evaluate and plot function at. Defaults to nodes of mesh
            function is defined on if :py:const:`None`.
        axis_size: Size of axis to plot each function on in inches as `(width, height)`
            tuple.
        arrangement: One of :py:const:`"horizontal"`, :py:const:`"vertical"` or
            :py:const:`"stacked"` corresponding to respectively plotting functions on
            separate axes in a single row, plotting functions on separate axes in a
            single column or plotting functions all on a single axis.

    Returns:
        Matplotlib figure object with plotted function(s).
    """
    label_and_functions = _preprocess_functions(functions)
    num_functions = len(label_and_functions)
    if arrangement == "vertical":
        n_rows, n_cols = num_functions, 1
        figsize = (axis_size[0], num_functions * axis_size[1])
    elif arrangement == "horizontal":
        n_rows, n_cols = 1, num_functions
        figsize = (num_functions * axis_size[0], axis_size[1])
    elif arrangement == "stacked":
        n_rows, n_cols = 1, 1
        figsize = axis_size
    else:
        msg = f"Value {arrangement} for arrangement invalid"
        raise ValueError(msg)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_1d(axes)
    for i, (label, function) in enumerate(label_and_functions):
        ax = axes[0] if arrangement == "stacked" else axes[i]
        mesh = function.function_space.mesh
        if mesh.topology.dim != 1:
            msg = "Only one-dimensional spatial domains are supported"
            raise ValueError(msg)
        points = mesh.geometry.x[:, 0] if points is None else points
        function_values = evaluate_function_at_points(function, points[:, None])
        ax.plot(points, function_values, label=label)
        ax.set(xlabel="Spatial coordinate", ylabel="Function value")
        if arrangement != "stacked":
            ax.set_title(label)
    if arrangement == "stacked":
        ax.legend()
    return fig


def plot_2d_functions(
    functions: Union[Function, list[Function], dict[str, Function]],
    *,
    plot_type: Literal["pcolor", "surface"] = "pcolor",
    axis_size: tuple[float, float] = (5.0, 5.0),
    colormap: Union[str, Colormap, None] = None,
    show_colorbar: bool = True,
    triangulation_color: Union[str, tuple[float, float, float], None] = None,
    arrangement: Literal["horizontal", "vertical"] = "horizontal",
) -> plt.Figure:
    """
    Plot one or more finite element functions on 2D domains using Matplotlib.

    Can plot either pseudocolor plots (heatmaps) for each function showing variation
    in function value by color mapping or a three-dimensional surface plot with
    height coordinate corresponding to function value.

    Args:
        functions: A single finite element function, sequence of functions or dictionary
            mapping from string labels to finite element functions, in all cases
            corresponding to the function(s) to plot. If a single function or sequence
            of functions are specified the function :py:attr:`name` attribute(s) will be
            used to set the title for each axis.
        plot_type: String specifying type of plot to use for each function: Either
            :py:const:`"pcolor"` for a pseudo color plot with function value represented
            by color, or :py:const:`"surface"` for a surface plot with function value
            represented by surface height.
        axis_size: Size of axis to plot each function on in inches as `(width, height)`
            tuple.
        colormap: Matplotlib colormap to use to plot function values (if
            :py:const:`None` default colormap is used).
        show_colorbar: Whether to show a colorbar key showing color mapping for function
            values.
        triangulation_color: If not :py:const:`None`, specifies the color (either as a
            string or RGB tuple) to use to plot the mesh triangulation as an overlay on
            heatmap.
        arrangement: Whether to arrange multiple axes vertically in a single column
            rather than default of horizontally in a single row.

    Returns:
        Matplotlib figure object with plotted function(s).
    """
    label_and_functions = _preprocess_functions(functions)
    num_functions = len(label_and_functions)
    multiplier = 1.25 if show_colorbar else 1.0
    if arrangement == "vertical":
        n_rows, n_cols = num_functions, 1
        figsize = (multiplier * axis_size[0], num_functions * axis_size[1])
    elif arrangement == "horizontal":
        n_rows, n_cols = 1, num_functions
        figsize = (multiplier * num_functions * axis_size[0], axis_size[1])
    else:
        msg = f"Value of arrangement argument {arrangement} is invalid"
        raise ValueError(msg)
    subplot_kw = {"projection": "3d"} if plot_type == "surface" else {}
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, subplot_kw=subplot_kw)
    for ax, (label, function) in zip(np.atleast_1d(axes), label_and_functions):
        mesh = function.function_space.mesh
        if mesh.topology.dim != 2:
            msg = "Only two-dimensional spatial domains are supported"
            raise ValueError(msg)
        triangulation = get_matplotlib_triangulation_from_mesh(mesh)
        function_values = evaluate_function_at_points(function, mesh.geometry.x)
        if plot_type == "surface":
            artist = ax.plot_trisurf(
                triangulation,
                function_values,
                cmap=colormap,
                shade=False,
                edgecolor=triangulation_color,
                linewidth=None if triangulation_color is None else 0.2,
            )
        elif plot_type == "pcolor":
            artist = ax.tripcolor(
                triangulation,
                function_values,
                shading="gouraud",
                cmap=colormap,
            )
            if triangulation_color is not None:
                ax.triplot(triangulation, color=triangulation_color, linewidth=1.0)
        else:
            msg = f"Invalid plot_type argument {plot_type}"
            raise ValueError(msg)
        ax.set(
            xlabel="Spatial coordinate 0",
            ylabel="Spatial coordinate 1",
            title=label,
        )
        if show_colorbar:
            fig.colorbar(artist, ax=ax)
    fig.tight_layout()
    return fig


def define_dirichlet_boundary_condition(
    boundary_value: Union[Function, Constant, float],
    function_space: Optional[FunctionSpace] = None,
    *,
    boundary_indicator_function: Optional[
        Callable[[ufl.SpatialCoordinate], bool]
    ] = None,
) -> DirichletBCMetaClass:
    """
    Define DOLFINx object representing Dirichlet boundary condition.

    Args:
        boundary_value: Fixed value(s) to enforce at domain boundary, either as a single
            floating point (or :py:class:`dolfinx.fem.Constant`) value or a finite
            element function object which gives the required values when evaluated at
            the boundary degrees of freedom.
        function_space: Argument specifying finite element function space from which
            boundary degrees of freedom should be computed. If :py:obj:`boundary_values`
            is a :py:class:`dolfinx.fem.Function` instance then should be set to
            :py:const:`None` (the default) as in this case
            :py:attr:`boundary_values.function_space` will be used as the relevant
            function space.
        boundary_indicator_function: If specified, a function evaluating to
            :py:const:`True` when the passed spatial coordinate is on the boundary and
            :py:const:`False` otherwise. If not specified (the default) then the
            boundary is assumed to correspond to all exterior facets.

    Returns:
        Dirichlet boundary condition object.
    """
    if function_space is None and isinstance(boundary_value, Function):
        function_space = boundary_value.function_space
    elif function_space is not None and isinstance(boundary_value, Function):
        msg = "function_space must be None if boundary_value is a Function"
        raise ValueError(msg)
    elif function_space is None:
        msg = "function_space must not be None if boundary_value is not a Function"
        raise ValueError(msg)
    mesh = function_space.mesh
    if boundary_indicator_function is not None:
        boundary_dofs = dolfinx.fem.locate_dofs_geometrical(
            function_space,
            boundary_indicator_function,
        )
    else:
        facet_dim = mesh.topology.dim - 1
        mesh.topology.create_connectivity(facet_dim, mesh.topology.dim)
        boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
        boundary_dofs = dolfinx.fem.locate_dofs_topological(
            function_space,
            facet_dim,
            boundary_facets,
        )
    return dolfinx.fem.dirichletbc(
        boundary_value,
        boundary_dofs,
        function_space if not isinstance(boundary_value, Function) else None,
    )
