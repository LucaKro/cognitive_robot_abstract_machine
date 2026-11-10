"""
Tests for choosing the open-vocabulary subtrees for a GPSR query.
"""

from __future__ import annotations

# %% Imports
import pytest
from robokudo.annotators.gpsr.gpsr_results import QueryType
from robokudo.descriptors.analysis_engines.gpsr.gpsr_implementation import GpsrRoute
from typing_extensions import TYPE_CHECKING

from ._gpsr_routing import gpsr_query

if TYPE_CHECKING:
    from robokudo.tree_components.query_based_task_scheduler import QueryBasedScheduler
    from robokudo_msgs.action import Query


# %% open-vocabulary routes
@pytest.mark.parametrize(
    "query, expected_route",
    [
        (
            gpsr_query(QueryType.DOOR_OPEN_VOCABULARY),
            GpsrRoute.DOOR_OPEN_VOCABULARY.value,
        ),
        (
            gpsr_query("human", color=["red"], description=["shirt"]),
            GpsrRoute.OPEN_VOCABULARY.value,
        ),
        (
            gpsr_query(
                "human", color=["red"], description=["shirt"], location="living_room"
            ),
            GpsrRoute.OPEN_VOCABULARY_IN_ROOM.value,
        ),
    ],
)
def test_query_selects_a_registered_open_vocabulary_route(
    scheduler: QueryBasedScheduler, query: Query.Goal, expected_route: str
) -> None:
    """
    Select an available route for a query answered by open-vocabulary detection.
    """
    assert scheduler.filter_fn(query) == expected_route
    assert expected_route in scheduler.tasks
