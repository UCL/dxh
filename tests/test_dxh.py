"""Tests for DOLFINx helpers (dxh) module."""

import dolfinx
import matplotlib.pyplot as plt
import numpy as np
import pytest
import ufl
from matplotlib.tri import Triangulation
from mpi4py import MPI

import dxh


@pytest.mark.parametrize("number_cells_per_axis", [5, 13, 20])
def test_get_matplotlib_triangulation_from_mesh(number_cells_per_axis):
    mesh = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD,
        *(number_cells_per_axis,) * 2,
    )
    triangulation = dxh.get_matplotlib_triangulation_from_mesh(mesh)
    assert isinstance(triangulation, Triangulation)
    assert triangulation.x.shape == ((number_cells_per_axis + 1) ** 2,)
    assert triangulation.y.shape == ((number_cells_per_axis + 1) ** 2,)


def _check_projected_expression(
    projected_expression,
    function_space,
    expression_function,
    number_cells_per_axis,
    convergence_order,
    convergence_constant,
):
    mesh = function_space.mesh
    assert isinstance(projected_expression, dolfinx.fem.Function)
    assert projected_expression.function_space == function_space
    assert (
        np.mean(
            (
                dxh.evaluate_function_at_points(projected_expression, mesh.geometry.x)
                - expression_function(mesh.geometry.x.T).T
            )
            ** 2,
        )
        ** 0.5
        < convergence_constant / number_cells_per_axis**convergence_order
    )


def _create_unit_mesh(spatial_dimension, number_cells_per_axis):
    if spatial_dimension == 1:
        return dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, number_cells_per_axis)
    elif spatial_dimension == 2:
        return dolfinx.mesh.create_unit_square(
            MPI.COMM_WORLD,
            number_cells_per_axis,
            number_cells_per_axis,
        )
    elif spatial_dimension == 3:
        return dolfinx.mesh.create_unit_cube(
            MPI.COMM_WORLD,
            number_cells_per_axis,
            number_cells_per_axis,
            number_cells_per_axis,
        )
    else:
        msg = f"Invalid spatial dimension: {spatial_dimension}"
        raise ValueError(msg)


def _one_dimensional_linear(spatial_coordinate):
    return 2.0 * spatial_coordinate[0] + 1.0


def _two_dimensional_linear(spatial_coordinate):
    return 3.0 * spatial_coordinate[0] + 2 * spatial_coordinate[1] - 2.0


def _three_dimensional_linear(spatial_coordinate):
    return (
        4.0 * spatial_coordinate[0]
        - 3.0 * spatial_coordinate[1]
        + 2.0 * spatial_coordinate[0]
        - 1.0
    )


def _one_dimensional_quadratic(spatial_coordinate):
    return spatial_coordinate[0] ** 2 + spatial_coordinate[0] + 1.0


def _two_dimensional_quadratic(spatial_coordinate):
    return (
        spatial_coordinate[0] ** 2 + spatial_coordinate[0] * spatial_coordinate[1] + 1.0
    )


def _three_dimensional_quadratic(spatial_coordinate):
    return spatial_coordinate[0] * spatial_coordinate[1] * spatial_coordinate[2] + 1.0


@pytest.mark.parametrize("number_cells_per_axis", [3, 10, 30, 100])
@pytest.mark.parametrize(
    "dimension_and_expression_function",
    [
        (1, _one_dimensional_quadratic),
        (2, _two_dimensional_quadratic),
        (3, _three_dimensional_quadratic),
        (1, _one_dimensional_linear),
        (2, _two_dimensional_linear),
        (3, _three_dimensional_linear),
    ],
)
@pytest.mark.parametrize("order", [1, 2])
def test_project_expression_on_to_function_space(
    number_cells_per_axis,
    dimension_and_expression_function,
    order,
):
    spatial_dimension, expression_function = dimension_and_expression_function
    if spatial_dimension == 3 and number_cells_per_axis > 30:
        pytest.skip("Skipping 3D spatial domain test with fine mesh resolution")
    mesh = _create_unit_mesh(spatial_dimension, number_cells_per_axis)
    function_space = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", order))
    convergence_order = order + 1
    convergence_constant = 3.0
    _check_projected_expression(
        dxh.project_expression_on_function_space(
            expression_function(ufl.SpatialCoordinate(mesh)),
            function_space,
        ),
        function_space,
        expression_function,
        number_cells_per_axis,
        convergence_order,
        convergence_constant,
    )
    _check_projected_expression(
        dxh.project_expression_on_function_space(
            expression_function,
            function_space,
        ),
        function_space,
        expression_function,
        number_cells_per_axis,
        convergence_order,
        convergence_constant,
    )


@pytest.mark.parametrize("number_cells_per_axis", [3, 10])
@pytest.mark.parametrize(
    "dimension_and_expression_function",
    [
        (1, _one_dimensional_quadratic),
        (2, _two_dimensional_quadratic),
        (3, _three_dimensional_quadratic),
    ],
)
@pytest.mark.parametrize("order", [1, 2])
def test_evaluate_function_at_points(
    number_cells_per_axis,
    dimension_and_expression_function,
    order,
):
    spatial_dimension, expression_function = dimension_and_expression_function
    mesh = _create_unit_mesh(spatial_dimension, number_cells_per_axis)
    function_space = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", order))
    function = dolfinx.fem.Function(function_space)
    function.interpolate(expression_function)
    assert np.allclose(
        dxh.evaluate_function_at_points(function, mesh.geometry.x),
        expression_function(mesh.geometry.x.T).T,
    )


@pytest.mark.parametrize("number_cells_per_axis", [3, 10])
@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("arrangement", ["vertical", "horizontal", "stacked"])
def test_plot_1d_functions(number_cells_per_axis, order, arrangement):
    mesh = _create_unit_mesh(1, number_cells_per_axis)
    function_space = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", order))
    functions_dict = {}
    for f in (_one_dimensional_linear, _one_dimensional_quadratic):
        function = dolfinx.fem.Function(function_space)
        function.interpolate(f)
        functions_dict[f.__name__[1:]] = function
    for functions_argument in (
        functions_dict,
        list(functions_dict.values()),
        next(iter(functions_dict.values())),
    ):
        fig = dxh.plot_1d_functions(functions_argument, arrangement=arrangement)
        assert isinstance(fig, plt.Figure)
        number_functions = (
            1
            if isinstance(functions_argument, dolfinx.fem.Function)
            or arrangement == "stacked"
            else len(functions_argument)
        )
        assert len(fig.get_axes()) == number_functions
        plt.close(fig)


@pytest.mark.parametrize("number_cells_per_axis", [3, 10])
@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("plot_type", ["pcolor", "surface"])
@pytest.mark.parametrize("colormap", [None, "magma"])
@pytest.mark.parametrize("show_colorbar", [True, False])
@pytest.mark.parametrize(
    "triangulation_color",
    [None, "white", "#fff", (1.0, 1.0, 1.0)],
)
@pytest.mark.parametrize("arrangement", ["horizontal", "vertical"])
def test_plot_2d_functions(
    number_cells_per_axis,
    order,
    plot_type,
    colormap,
    show_colorbar,
    triangulation_color,
    arrangement,
):
    mesh = _create_unit_mesh(2, number_cells_per_axis)
    function_space = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", order))
    functions_dict = {}
    for f in (_two_dimensional_linear, _two_dimensional_quadratic):
        function = dolfinx.fem.Function(function_space)
        function.interpolate(f)
        functions_dict[f.__name__[1:]] = function
    for functions_argument in (
        functions_dict,
        list(functions_dict.values()),
        next(iter(functions_dict.values())),
    ):
        fig = dxh.plot_2d_functions(
            functions_argument,
            show_colorbar=show_colorbar,
            plot_type=plot_type,
            colormap=colormap,
            triangulation_color=triangulation_color,
            arrangement=arrangement,
        )
        assert isinstance(fig, plt.Figure)
        number_functions = (
            1
            if isinstance(functions_argument, dolfinx.fem.Function)
            else len(functions_argument)
        ) * (2 if show_colorbar else 1)
        assert len(fig.get_axes()) == number_functions
        plt.close(fig)
