"""
Tests for choosing the face and the vision language subtree for a GPSR query.
"""

from __future__ import annotations

# %% Imports
import pytest
from robokudo.descriptors.analysis_engines.gpsr.gpsr_implementation import GpsrRoute
from typing_extensions import TYPE_CHECKING

from ._gpsr_routing import gpsr_query

if TYPE_CHECKING:
    from robokudo.tree_components.query_based_task_scheduler import QueryBasedScheduler
    from robokudo_msgs.action import Query


# %% optional routes
@pytest.mark.parametrize(
    "query, expected_route",
    [
        (gpsr_query("face"), GpsrRoute.FACE.value),
        (gpsr_query("human", ["Attribute"]), GpsrRoute.HUMAN_ATTRIBUTE.value),
    ],
    ids=["face recognition", "person description"],
)
def test_query_selects_a_registered_optional_route(
    scheduler: QueryBasedScheduler, query: Query.Goal, expected_route: str
) -> None:
    """
    Select a registered route for face recognition and open descriptions.
    """
    assert scheduler.filter_fn(query) == expected_route
    assert expected_route in scheduler.tasks
