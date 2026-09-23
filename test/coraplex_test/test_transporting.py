"""
What a transport does about an object it finds inside a container.
"""

from coraplex.datastructures.enums import Arms, ReachFraction
from coraplex.locations import factories
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.composite.transporting import (
    MoveAndPickUpAction,
    TransportAction,
)
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Drawer,
    Handle,
    Milk,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose

# %% fetching an object out of a drawer

DRAWER = "cabinet10_drawer_top"
"""
The apartment drawer the transport opens on its way to the object inside it.
"""

DRAWER_HANDLE = "handle_cab10_t"
"""
The handle of :data:`DRAWER`.
"""


def test_opening_a_container_on_the_way_stands_where_it_is_opened_from(
    mutable_model_world, monkeypatch
):
    """
    An object inside a drawer is fetched by opening the drawer first, and the robot
    stands back for that the way it does for any container rather than as close as it
    would to grasp something that stays put.
    """
    world, robot, context = mutable_model_world
    drawer_body = world.get_body_by_name(DRAWER)
    with world.modify_world():
        world.add_semantic_annotation_recursively(
            Drawer(
                root=drawer_body,
                handle=Handle(root=world.get_body_by_name(DRAWER_HANDLE)),
            )
        )
    transport = TransportAction(
        grasp=world.get_semantic_annotations_by_type(Milk)[0].grasp_poses()[0],
        target_location=Pose(reference_frame=world.root),
        arm=Arms.RIGHT,
    )
    sequential([transport], context)

    asked_for = {}
    build_location = factories.reachability_location
    monkeypatch.setattr(
        factories,
        "reachability_location",
        lambda *args, **kwargs: asked_for.update(kwargs)
        or build_location(*args, **kwargs),
    )

    transport._make_open_container_actions(drawer_body)

    assert asked_for["reach_fraction"] == ReachFraction.ACCESSING


# %% moving to an object and picking it up


def test_move_and_pick_up_takes_the_grasp_it_was_given(mutable_model_world):
    """
    The caller chooses the grasp, so the pick-up at the end of the walk takes that one
    rather than whichever grasp the object happens to list first.
    """
    world, robot, context = mutable_model_world
    grasp = world.get_semantic_annotations_by_type(Milk)[0].grasp_poses()[-1]
    move_and_pick_up = MoveAndPickUpAction(
        standing_position=Pose(reference_frame=world.root),
        grasp=grasp,
        arm=Arms.LEFT,
    )
    sequential([move_and_pick_up], context)

    pick_ups = [
        child
        for child in move_and_pick_up._action_plan.children
        if isinstance(getattr(child, "designator", None), PickUpAction)
    ]

    assert [pick_up.designator.grasp for pick_up in pick_ups] == [grasp]
