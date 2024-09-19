"""Tests for DOLFINx helpers (dxh) module."""

import dolfinx
import matplotlib.pyplot as plt
import numpy as np
import pytest
import ufl
from matplotlib.tri import Triangulation
from mpi4py import MPI

import dxh

try:
    from dolfinx.fem import DirichletBC
except ImportError:
    # Compatibility w dolfinx@0.6: try importing old DirichletBCMetaClass name.
    from dolfinx.fem import DirichletBCMetaClass as DirichletBC

try:
    from dolfinx.fem import functionspace
except ImportError:
    # Compatibility w dolfinx@0.6: if the new functionspace function is not in DOLFINx
    # then use the class constructor directly.
    from dolfinx.fem import FunctionSpace as functionspace  # noqa: N813


try:
    from basix.ufl import element as basix_element
    from basix.ufl import mixed_element as basix_mixed_element
except ImportError:
    # Compatibility w dolfinx@0.6 / basix<0.7: try import from previous
    # basix.ufl_wrapper submodule if basix.ufl not found
    from basix.ufl_wrapper import (
        MixedElement as basix_mixed_element,  # noqa: N813
    )
    from basix.ufl_wrapper import (
        create_element as basix_element,
    )


def _create_unit_mesh(spatial_dimension, number_cells_per_axis, cell_type=None):
    if spatial_dimension == 1:
        return dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, number_cells_per_axis)
    elif spatial_dimension == 2:
        cell_type = dolfinx.mesh.CellType.triangle if cell_type is None else cell_type
        return dolfinx.mesh.create_unit_square(
            MPI.COMM_WORLD,
            number_cells_per_axis,
            number_cells_per_axis,
            cell_type=cell_type,
        )
    elif spatial_dimension == 3:
        cell_type = (
            dolfinx.mesh.CellType.tetrahedron if cell_type is None else cell_type
        )
        return dolfinx.mesh.create_unit_cube(
            MPI.COMM_WORLD,
            number_cells_per_axis,
            number_cells_per_axis,
            number_cells_per_axis,
            cell_type=cell_type,
        )
    else:
        msg = f"Invalid spatial dimension: {spatial_dimension}"
        raise ValueError(msg)


@pytest.mark.parametrize("number_cells_per_axis", [5, 13, 20])
def test_get_matplotlib_triangulation_from_mesh(number_cells_per_axis):
    mesh = _create_unit_mesh(2, number_cells_per_axis)
    triangulation = dxh.get_matplotlib_triangulation_from_mesh(mesh)
    assert isinstance(triangulation, Triangulation)
    assert triangulation.x.shape == ((number_cells_per_axis + 1) ** 2,)
    assert triangulation.y.shape == ((number_cells_per_axis + 1) ** 2,)


def test_get_matplotlib_triangulation_from_mesh_invalid_dimension():
    mesh = _create_unit_mesh(1, 5)
    with pytest.raises(ValueError, match="two-dimensional"):
        dxh.get_matplotlib_triangulation_from_mesh(mesh)


def test_get_matplotlib_triangulation_from_mesh_invalid_cell_type():
    mesh = _create_unit_mesh(2, 5, cell_type=dolfinx.mesh.CellType.quadrilateral)
    with pytest.raises(ValueError, match="triangular"):
        dxh.get_matplotlib_triangulation_from_mesh(mesh)


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
@pytest.mark.parametrize("degree", [1, 2])
def test_project_expression_on_to_function_space(
    number_cells_per_axis,
    dimension_and_expression_function,
    degree,
):
    spatial_dimension, expression_function = dimension_and_expression_function
    if spatial_dimension == 3 and number_cells_per_axis > 30:
        pytest.skip("Skipping 3D spatial domain test with fine mesh resolution")
    mesh = _create_unit_mesh(spatial_dimension, number_cells_per_axis)
    function_space = functionspace(mesh, ("Lagrange", degree))
    convergence_order = degree + 1
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
@pytest.mark.parametrize("degree", [1, 2])
def test_evaluate_function_at_points(
    number_cells_per_axis,
    dimension_and_expression_function,
    degree,
):
    spatial_dimension, expression_function = dimension_and_expression_function
    mesh = _create_unit_mesh(spatial_dimension, number_cells_per_axis)
    function_space = functionspace(mesh, ("Lagrange", degree))
    function = dolfinx.fem.Function(function_space)
    function.interpolate(expression_function)
    assert np.allclose(
        dxh.evaluate_function_at_points(function, mesh.geometry.x),
        expression_function(mesh.geometry.x.T).T,
    )


@pytest.mark.parametrize(
    "dimension_and_expression_function",
    [
        (1, _one_dimensional_quadratic),
        (2, _two_dimensional_quadratic),
        (3, _three_dimensional_quadratic),
    ],
)
def test_evaluate_function_at_points_outside_domain(dimension_and_expression_function):
    spatial_dimension, expression_function = dimension_and_expression_function
    mesh = _create_unit_mesh(spatial_dimension, 5)
    function_space = functionspace(mesh, ("Lagrange", 1))
    function = dolfinx.fem.Function(function_space)
    function.interpolate(expression_function)
    with pytest.raises(ValueError, match="domain"):
        dxh.evaluate_function_at_points(function, np.full(spatial_dimension, -1.0))


def test_evaluate_function_at_points_invalid_points():
    spatial_dimension, expression_function = 1, _one_dimensional_quadratic
    mesh = _create_unit_mesh(spatial_dimension, 5)
    function_space = functionspace(mesh, ("Lagrange", 1))
    function = dolfinx.fem.Function(function_space)
    function.interpolate(expression_function)
    with pytest.raises(ValueError, match="points"):
        dxh.evaluate_function_at_points(function, np.ones((1, 1, 1)))
    with pytest.raises(ValueError, match="points"):
        dxh.evaluate_function_at_points(function, np.ones(2))


def _interpolate_functions(function_space, functions):
    functions_dict = {}
    for f in functions:
        function = dolfinx.fem.Function(function_space)
        function.interpolate(f)
        functions_dict[f.__name__[1:]] = function
    return functions_dict


@pytest.mark.parametrize("number_cells_per_axis", [3, 10])
@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("points", [None, np.linspace(0, 1, 3)])
@pytest.mark.parametrize("arrangement", ["vertical", "horizontal", "stacked"])
@pytest.mark.parametrize("share_value_axis", [True, False])
def test_plot_1d_functions(
    number_cells_per_axis,
    points,
    degree,
    arrangement,
    share_value_axis,
):
    mesh = _create_unit_mesh(1, number_cells_per_axis)
    function_space = functionspace(mesh, ("Lagrange", degree))
    functions_dict = _interpolate_functions(
        function_space,
        (_one_dimensional_linear, _one_dimensional_quadratic),
    )
    for functions_argument in (
        functions_dict,
        list(functions_dict.values()),
        next(iter(functions_dict.values())),
    ):
        fig = dxh.plot_1d_functions(
            functions_argument,
            points=points,
            arrangement=arrangement,
            share_value_axis=share_value_axis,
        )
        assert isinstance(fig, plt.Figure)
        number_functions = (
            1
            if isinstance(functions_argument, dolfinx.fem.Function)
            or arrangement == "stacked"
            else len(functions_argument)
        )
        assert len(fig.get_axes()) == number_functions
        plt.close(fig)


def test_plot_1d_functions_invalid_arrangement():
    mesh = _create_unit_mesh(1, 5)
    function_space = functionspace(mesh, ("Lagrange", 1))
    functions_dict = _interpolate_functions(
        function_space,
        (_one_dimensional_linear, _one_dimensional_quadratic),
    )
    with pytest.raises(ValueError, match="arrangement"):
        dxh.plot_1d_functions(functions_dict, arrangement="invalid")


def test_plot_1d_functions_invalid_dimension():
    mesh = _create_unit_mesh(2, 5)
    function_space = functionspace(mesh, ("Lagrange", 1))
    functions_dict = _interpolate_functions(
        function_space,
        (_two_dimensional_linear, _two_dimensional_quadratic),
    )
    with pytest.raises(ValueError, match="dimension"):
        dxh.plot_1d_functions(functions_dict)


@pytest.mark.parametrize("number_cells_per_axis", [3, 10])
@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("plot_type", ["pcolor", "surface"])
@pytest.mark.parametrize("colormap", [None, "magma"])
@pytest.mark.parametrize("show_colorbar", [True, False])
@pytest.mark.parametrize(
    "triangulation_color",
    [None, "white", "#fff", (1.0, 1.0, 1.0)],
)
@pytest.mark.parametrize("arrangement", ["horizontal", "vertical"])
@pytest.mark.parametrize("share_value_axis", [True, False])
def test_plot_2d_functions(
    number_cells_per_axis,
    degree,
    plot_type,
    colormap,
    show_colorbar,
    triangulation_color,
    arrangement,
    share_value_axis,
):
    mesh = _create_unit_mesh(2, number_cells_per_axis)
    function_space = functionspace(mesh, ("Lagrange", degree))
    functions_dict = _interpolate_functions(
        function_space,
        (_two_dimensional_linear, _two_dimensional_quadratic),
    )
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
            share_value_axis=share_value_axis,
        )
        assert isinstance(fig, plt.Figure)
        number_functions = (
            1
            if isinstance(functions_argument, dolfinx.fem.Function)
            else len(functions_argument)
        )
        expected_number_axes = (
            2 * number_functions if show_colorbar else number_functions
        )
        assert len(fig.get_axes()) == expected_number_axes
        plt.close(fig)


def test_plot_2d_functions_invalid_arrangement():
    mesh = _create_unit_mesh(2, 5)
    function_space = functionspace(mesh, ("Lagrange", 1))
    functions_dict = _interpolate_functions(
        function_space,
        (_two_dimensional_linear, _two_dimensional_quadratic),
    )
    with pytest.raises(ValueError, match="arrangement"):
        dxh.plot_2d_functions(functions_dict, arrangement="invalid")


def test_plot_2d_functions_invalid_plot_type():
    mesh = _create_unit_mesh(2, 5)
    function_space = functionspace(mesh, ("Lagrange", 1))
    functions_dict = _interpolate_functions(
        function_space,
        (_two_dimensional_linear, _two_dimensional_quadratic),
    )
    with pytest.raises(ValueError, match="plot_type"):
        dxh.plot_2d_functions(functions_dict, plot_type="invalid")


def test_plot_2d_functions_invalid_dimension():
    mesh = _create_unit_mesh(1, 5)
    function_space = functionspace(mesh, ("Lagrange", 1))
    functions_dict = _interpolate_functions(
        function_space,
        (_one_dimensional_linear, _one_dimensional_quadratic),
    )
    with pytest.raises(ValueError, match="dimension"):
        dxh.plot_2d_functions(functions_dict)


def _unit_mesh_boundary_indicator_function(spatial_coordinate):
    return ((spatial_coordinate == 0) | (spatial_coordinate == 1)).any(axis=0)


def _zero_vector(vector):
    """Fill the vector with zeros.

    Accounts for the dolfinx 0.7 and 0.6 API differences.

    Todo:
        Remove this function and use `vector.array.fill(0.0)` directly in the
        code when dropping support for 0.6.
    """
    try:
        vector.array.fill(0.0)  # dolfinx 0.7: underlying vector is numpy.ndarray.
    except AttributeError:
        vector.set(0.0)  # dolfinx 0.6: underlying vector is _cpp DOLFINx vector.


@pytest.mark.parametrize("number_cells_per_axis", [3, 10])
@pytest.mark.parametrize("spatial_dimension", [1, 2, 3])
@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("boundary_value_type", ["function", "constant", "float"])
@pytest.mark.parametrize(
    "boundary_indicator_function",
    [None, _unit_mesh_boundary_indicator_function],
)
def test_define_dirichlet_boundary_condition(
    number_cells_per_axis,
    spatial_dimension,
    degree,
    boundary_value_type,
    boundary_indicator_function,
):
    mesh = _create_unit_mesh(spatial_dimension, number_cells_per_axis)
    function_space = functionspace(mesh, ("Lagrange", degree))
    if boundary_value_type == "function":
        boundary_value = dolfinx.fem.Function(function_space)
        _zero_vector(boundary_value.x)
    elif boundary_value_type == "constant":
        boundary_value = dolfinx.fem.Constant(mesh, 0.0)
    elif boundary_value_type == "float":
        boundary_value = 0.0
    else:
        msg = f"Unexpected boundary_value_type: {boundary_value_type}"
        raise ValueError(msg)
    boundary_condition = dxh.define_dirichlet_boundary_condition(
        boundary_value,
        boundary_indicator_function=boundary_indicator_function,
        function_space=None if boundary_value_type == "function" else function_space,
    )
    assert isinstance(boundary_condition, DirichletBC)
    assert boundary_condition.function_space == function_space._cpp_object


def test_define_dirichlet_boundary_condition_missing_function_space():
    with pytest.raises(ValueError, match="function_space"):
        dxh.define_dirichlet_boundary_condition(0.0)


def test_define_dirichlet_boundary_condition_function_with_function_space():
    mesh = _create_unit_mesh(1, 5)
    function_space = functionspace(mesh, ("Lagrange", 1))
    boundary_value = dolfinx.fem.Function(function_space)
    with pytest.raises(ValueError, match="function_space"):
        dxh.define_dirichlet_boundary_condition(boundary_value, function_space)


@pytest.mark.parametrize("number_cells_per_axis", [3, 10])
@pytest.mark.parametrize("spatial_dimension", [1, 2, 3])
@pytest.mark.parametrize("element_degrees", [[1], [2, 3], [1, 2, 3]])
@pytest.mark.parametrize(
    "boundary_indicator_function",
    [None, _unit_mesh_boundary_indicator_function],
)
def test_define_dirichlet_boundary_conditions_on_mixed_space(
    number_cells_per_axis,
    spatial_dimension,
    element_degrees,
    boundary_indicator_function,
):
    mesh = _create_unit_mesh(spatial_dimension, number_cells_per_axis)
    elements = [
        # Compatibility w dolfinx@0.6: Use mesh.ufl_cell().cellname() rather
        # than mesh.basix_cell()
        basix_element("Lagrange", mesh.ufl_cell().cellname(), d)
        for d in element_degrees
    ]
    mixed_element = basix_mixed_element(elements)
    mixed_function_space = functionspace(mesh, mixed_element)
    boundary_values = [0.0] * len(element_degrees)
    boundary_indicator_functions = [boundary_indicator_function] * len(element_degrees)
    boundary_conditions = dxh.define_dirichlet_boundary_conditions_on_mixed_space(
        boundary_values,
        mixed_function_space,
        boundary_indicator_functions=boundary_indicator_functions,
    )
    assert all(isinstance(bc, DirichletBC) for bc in boundary_conditions)


@pytest.mark.parametrize("number_cells_per_axis", [10, 30, 100])
@pytest.mark.parametrize(
    "dimension_and_expression_function",
    [
        (1, _one_dimensional_quadratic),
        (2, _two_dimensional_quadratic),
        (3, _three_dimensional_quadratic),
    ],
)
@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("degree_raise", [0, 1, 3])
@pytest.mark.parametrize("norm_order", [1, 2, "inf-dof"])
def test_error_norm(
    number_cells_per_axis,
    dimension_and_expression_function,
    degree,
    degree_raise,
    norm_order,
):
    spatial_dimension, expression_function = dimension_and_expression_function
    if spatial_dimension == 3 and number_cells_per_axis > 30:
        pytest.skip("Skipping 3D spatial domain test with fine mesh resolution")
    mesh = _create_unit_mesh(spatial_dimension, number_cells_per_axis)
    function_space = functionspace(mesh, ("Lagrange", degree))
    function_1 = dolfinx.fem.Function(function_space)
    function_1.interpolate(expression_function)
    spatial_coordinate = ufl.SpatialCoordinate(mesh)
    expression = expression_function(spatial_coordinate)
    for function_or_expression_2 in (expression_function, expression, function_1):
        error = dxh.error_norm(
            function_1,
            function_or_expression_2,
            degree_raise,
            norm_order,
        )
        assert isinstance(error, float)
        assert error >= 0
        # We expect computed norms to be close to zero other than approximation error -
        # we assume here approximation error is O(h^(degree+1)) in mesh size h.
        assert error < 1.0 / number_cells_per_axis ** (degree + 1)


def test_error_norm_with_invalid_norm_order():
    mesh = _create_unit_mesh(1, 3)
    function_space = functionspace(mesh, ("Lagrange", 1))
    function = dolfinx.fem.Function(function_space)
    function.interpolate(_one_dimensional_linear)
    with pytest.raises(ValueError, match="norm_order"):
        dxh.error_norm(function, function, norm_order=-1)
    with pytest.raises(ValueError, match="norm_order"):
        dxh.error_norm(function, function, norm_order=3)
    with pytest.raises(ValueError, match="norm_order"):
        dxh.error_norm(function, function, norm_order="a")
