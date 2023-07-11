"""Helper functions for dolfinx"""


from typing import Callable, Optional, Sequence, Union

import dolfinx
import matplotlib.pyplot as plt
import numpy as np
import ufl
from dolfinx.fem import Constant, DirichletBCMetaClass, Function, FunctionSpace
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import Mesh
from matplotlib.colors import Colormap
from matplotlib.tri import Triangulation
from numpy.typing import ArrayLike


def get_matplotlib_triangulation_from_mesh(mesh: Mesh) -> Triangulation:
    """Get matplotlib triangulation corresponding to dolfinx mesh.

    Args:
        mesh: Finite element mesh to get triangulation for.

    Returns:
        Object representing triangulation of mesh to use in Matplotlib plot functions.
    """
    assert mesh.topology.dim == 2, "Only two-dimensional spatial domains are supported"
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
        ufl.core.expr.Expr, Callable[[ufl.SpatialCoordinate], ufl.core.expr.Expr]
    ],
    function_space: ufl.FunctionSpace,
) -> Function:
    """Project expression onto finite element function space.

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


def evaluate_function_at_points(function: Function, points: ArrayLike) -> ArrayLike:
    """Evaluate a finite element function at one or more points.

    Args:
        function: Finite element function to evaluate.
        points: One or more points in domain of function to evaluate at. Should be
            either a one-dimensional array corresponding to a single point (with size
            equal to the geometric dimension or 3) or a two-dimensional  array
            corresponding to one point per row (with size of last axis equal to the
            geometric dimension or 3).

    Returns:
        Value(s) of function evaluated at point(s).
    """
    mesh = function.function_space.mesh
    assert points.ndim in (1, 2)
    if points.ndim == 1:
        points = points[None]
    assert points.shape[-1] in (3, mesh.geometry.dim)
    if points.shape[-1] != 3:
        padded_points = np.zeros(points.shape[:-1] + (3,))
        padded_points[..., : points.shape[-1]] = points
        points = padded_points
    tree = dolfinx.geometry.BoundingBoxTree(mesh, mesh.geometry.dim)
    cell_candidates = dolfinx.geometry.compute_collisions(tree, points)
    assert np.all(
        cell_candidates.offsets[1:] > 0
    ), "One or more points not within domain"
    cell_adjacency_list = dolfinx.geometry.compute_colliding_cells(
        mesh, cell_candidates, points
    )
    first_cell_indices = cell_adjacency_list.array[cell_adjacency_list.offsets[:-1]]
    return np.squeeze(function.eval(points, first_cell_indices))


def _preprocess_functions(
    functions: Union[Function, Sequence[Function], dict[str, Function]]
) -> list[tuple[str, Function]]:
    if isinstance(functions, Function):
        return [(functions.name, functions)]
    elif isinstance(functions, dict):
        return list(functions.items())
    else:
        return [(f.name, f) for f in functions]


def plot_1d_functions(
    functions: Union[Function, Sequence[Function], dict[str, Function]],
    points: Optional[ArrayLike] = None,
    axis_size: tuple[float, float] = (5.0, 5.0),
    arrangement: str = "horizontal",
) -> plt.Figure:
    """Plot one or more finite element functions on 1D domains using Matplotlib.

    Args:
        functions: A single finite element function, sequence of functions or dictionary
            mapping from string labels to finite element functions, in all cases
            corresponding to the function(s) to plot. If a single function or sequence
            of functions are specified the function `name` attribute(s) will be used to
            set the title for each axis.
        axis_size: Size of axis to plot each function on in inches as `(width, height)`
            tuple.
        arrangment: One of "horizontal", "vertical" or "stacked" corresponding to
            respectively plotting functions on separate axes in a single row, plotting
            functions on separate axes in a single column or plotting functions all on a
            single axis.

    Return:
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
        raise ValueError(f"Value {arrangement} for arrangment invalid")
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_1d(axes)
    for i, (label, function) in enumerate(label_and_functions):
        ax = axes[0] if arrangement == "stacked" else axes[i]
        mesh = function.function_space.mesh
        assert (
            mesh.topology.dim == 1
        ), "Only one-dimensional spatial domains are supported"
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
    plot_surfaces: bool = False,
    axis_size: tuple[float, float] = (5.0, 5.0),
    colormap: Union[str, Colormap, None] = None,
    show_colorbar: bool = True,
    triangulation_color: Union[str, tuple[float, float, float], None] = None,
    arrange_vertically: bool = False,
) -> plt.Figure:
    """Plot one or more finite element functions on 2D domains using Matplotlib.

    Can plot either pseudocolor plots (heatmaps) for each function showing variation
    in function value by color mapping or a three-dimensional surface plot with
    height coordinate corresponding to function value.

    Args:
        functions: A single finite element function, sequence of functions or dictionary
            mapping from string labels to finite element functions, in all cases
            corresponding to the function(s) to plot. If a single function or sequence
            of functions are specified the function `name` attribute(s) will be used to
            set the title for each axis.
        plot_surfaces: Whether to plot triangulated surfaces with function value as
            height coordinate, rather than pseudocolor plots.
        axis_size: Size of axis to plot each function on in inches as `(width, height)`
            tuple.
        colormap: Matplotlib colormap to use to plot function values (if `None` default
            colormap is used).
        show_colorbar: Whether to show a colorbar key showing color mapping for function
            values.
        triangulation_color: If not `None`, specifies the color (either as a string or
            RGB tuple) to use to plot the mesh triangulation as an overlay on heatmap.
        arrange_vertically: Whether to arrange multiple axes vertically in a single
            column rather than default of horizontally in a single row.

    Return:
        Matplotlib figure object with plotted function(s).
    """
    label_and_functions = _preprocess_functions(functions)
    num_functions = len(label_and_functions)
    multiplier = 1.25 if show_colorbar else 1.0
    if arrange_vertically:
        n_rows, n_cols = num_functions, 1
        figsize = (multiplier * axis_size[0], num_functions * axis_size[1])
    else:
        n_rows, n_cols = 1, num_functions
        figsize = (multiplier * num_functions * axis_size[0], axis_size[1])
    subplot_kw = {"projection": "3d"} if plot_surfaces else {}
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, subplot_kw=subplot_kw)
    for ax, (label, function) in zip(np.atleast_1d(axes), label_and_functions):
        mesh = function.function_space.mesh
        triangulation = get_matplotlib_triangulation_from_mesh(mesh)
        assert (
            mesh.topology.dim == 2
        ), "Only two-dimensional spatial domains are supported"
        function_values = evaluate_function_at_points(function, mesh.geometry.x)
        if plot_surfaces:
            artist = ax.plot_trisurf(
                triangulation,
                function_values,
                cmap=colormap,
                shade=False,
                edgecolor=triangulation_color,
                linewidth=None if triangulation_color is None else 0.2,
            )
        else:
            artist = ax.tripcolor(
                triangulation, function_values, shading="gouraud", cmap=colormap
            )
            if triangulation_color is not None:
                ax.triplot(triangulation, color=triangulation_color, linewidth=1.0)
        ax.set(
            xlabel="Spatial coordinate 0", ylabel="Spatial coordinate 1", title=label
        )
        if show_colorbar:
            fig.colorbar(artist, ax=ax)
    fig.tight_layout()
    return fig


def define_dirchlet_boundary_condition(
    boundary_value: Union[Function, Constant, float],
    boundary_indicator_function: Optional[
        Callable[[ufl.SpatialCoordinate], bool]
    ] = None,
    function_space: Optional[FunctionSpace] = None,
) -> DirichletBCMetaClass:
    """Define dolfinx object representing Dirichlet boundary condition.

    Args:
        boundary_value: Fixed value(s) to enforce at domain boundary, either as a single
            floating point (or `Constant`) value or a finite element function object
            which gives the required values when evaluated at the boundary degrees of
            freedom.
        boundary_indicator_function: If specified, a function evaluating to `True` when
            the passed spatial coordinate is on the boundary and `False` otherwise. If
            not specified (the default) then the boundary is assumed to correspond to
            all exterior facets.
        function_space: Optional argument specifying finite element function space from
            which boundary degrees of freedom should be computed. If `None` (default)
            then `boundary_values.function_space` is used (which will only work if
            `boundary_values` is a `Function` instance.

    Returns:
        Dirichlet boundary condition object.
    """
    if function_space is None:
        function_space = boundary_value.function_space
    mesh = function_space.mesh
    if boundary_indicator_function is not None:
        boundary_dofs = dolfinx.fem.locate_dofs_geometrical(
            function_space, boundary_indicator_function
        )
    else:
        facet_dim = mesh.topology.dim - 1
        mesh.topology.create_connectivity(facet_dim, mesh.topology.dim)
        boundary_facets = dolfinx.fem.exterior_facet_indices(mesh.topology)
        boundary_dofs = dolfinx.fem.locate_dofs_topological(
            function_space, facet_dim, boundary_facets
        )
    return dolfinx.fem.dirichletbc(boundary_value, boundary_dofs, function_space)
