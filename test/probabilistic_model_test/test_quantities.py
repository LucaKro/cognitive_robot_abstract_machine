import numpy as np
import pytest
from random_events.variable import Continuous

from probabilistic_model.exceptions import (
    RepeatedVariableError,
    VariableNotInQuantitiesError,
)
from probabilistic_model.quantities import Quantities

# %% the order the variables are laid out in


class TestQuantityOrder:
    """
    A variable's place in the layout is the row and column every array over it uses, so
    the order the variables are named in is the order they keep.
    """

    def test_a_variable_named_twice_is_rejected(self):
        """
        Only one of the two rows could ever be read back, so the other's value would be
        carried along and never answered with.
        """
        repeated = Continuous("x")
        with pytest.raises(RepeatedVariableError) as error:
            Quantities.of(repeated, repeated)
        assert error.value.variable == repeated

    def test_each_variable_keeps_the_row_it_was_named_in(self):
        first, second = Continuous("a"), Continuous("b")
        quantities = Quantities.of(first, second)
        assert quantities.index_of(first) == 0
        assert quantities.index_of(second) == 1
        assert len(quantities) == 2

    def test_the_order_named_is_kept_even_when_it_is_not_alphabetical(self):
        """
        A layout names its variables in the order its own domain uses — a pose is laid
        out ``x, y, z, roll, pitch, yaw`` — so the layout must not sort them.
        """
        later, earlier = Continuous("z"), Continuous("a")
        quantities = Quantities.of(later, earlier)
        assert quantities.variables == (later, earlier)
        assert quantities.index_of(later) == 0

    def test_a_variable_that_was_not_named_is_rejected(self):
        named = Continuous("a")
        quantities = Quantities.of(named)
        unnamed = Continuous("b")
        with pytest.raises(VariableNotInQuantitiesError) as error:
            quantities.index_of(unnamed)
        assert error.value.variable == unnamed
        assert error.value.quantities == [named]

    def test_membership_answers_without_raising(self):
        named = Continuous("a")
        quantities = Quantities.of(named)
        assert named in quantities
        assert Continuous("b") not in quantities

    def test_iterating_yields_the_variables_in_layout_order(self):
        first, second = Continuous("a"), Continuous("b")
        assert list(Quantities.of(first, second)) == [first, second]


# %% building arrays over those variables


class TestArrayLayout:
    """
    Every array is built from the variables rather than from row and column counts, so
    an array cannot describe a different set of variables than the layout it belongs to.
    """

    def test_a_vector_holds_each_variable_in_its_own_row(self):
        first, second = Continuous("a"), Continuous("b")
        quantities = Quantities.of(first, second)
        assert quantities.vector({second: 3.0}).tolist() == [0.0, 3.0]

    def test_a_matrix_reads_a_pair_as_row_then_column(self):
        first, second = Continuous("a"), Continuous("b")
        quantities = Quantities.of(first, second)
        assert quantities.matrix({(first, second): 5.0}).tolist() == [
            [0.0, 5.0],
            [0.0, 0.0],
        ]

    def test_a_symmetric_matrix_fills_a_pair_both_ways(self):
        """
        Two variables vary together by one number rather than two, so stating a pair
        once fills its mirror.
        """
        first, second = Continuous("a"), Continuous("b")
        quantities = Quantities.of(first, second)
        assert quantities.symmetric_matrix({(first, second): 5.0}).tolist() == [
            [0.0, 5.0],
            [5.0, 0.0],
        ]

    def test_building_a_vector_for_an_unknown_variable_is_rejected(self):
        quantities = Quantities.of(Continuous("a"))
        unnamed = Continuous("b")
        with pytest.raises(VariableNotInQuantitiesError) as error:
            quantities.vector({unnamed: 1.0})
        assert error.value.variable == unnamed

    def test_building_a_matrix_for_an_unknown_variable_is_rejected(self):
        known = Continuous("a")
        quantities = Quantities.of(known)
        unnamed = Continuous("b")
        with pytest.raises(VariableNotInQuantitiesError) as error:
            quantities.matrix({(known, unnamed): 1.0})
        assert error.value.variable == unnamed

    def test_unchanged_is_the_transition_that_moves_nothing(self):
        first, second = Continuous("a"), Continuous("b")
        quantities = Quantities.of(first, second)
        assert quantities.matrix(quantities.unchanged).tolist() == np.eye(2).tolist()
