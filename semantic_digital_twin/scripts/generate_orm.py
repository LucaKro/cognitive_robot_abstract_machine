# ----------------------------------------------------------------------------------------------------------------------
# This script generates the ORM classes for the semantic_digital_twin package.
# Dataclasses can be mapped automatically to the ORM model
# using the ORMatic library, they just have to be registered in the classes list.
# Classes that are self_mapped and explicitly_mapped are already mapped in the model.py file. Look there for more
# information on how to map them.
# ----------------------------------------------------------------------------------------------------------------------
from __future__ import annotations

import logging
from pathlib import Path

import trimesh

import semantic_digital_twin
import semantic_digital_twin.orm.model

import semantic_digital_twin.adapters.procthor.procthor_resolver
from krrood.adapters.json_serializer import SubclassJSONSerializer
from krrood.entity_query_language.predicate import SymbolicCallable
from krrood.ormatic.ormatic import ORMatic
from krrood.utils import recursive_subclasses
import semantic_digital_twin.reasoning.predicates
import semantic_digital_twin.reasoning.world_rdr.rules
from semantic_digital_twin.semantic_annotations.position_descriptions import (
    SemanticDirection,
)
from semantic_digital_twin.spatial_computations.forward_kinematics import (
    ForwardKinematicsManager,
)
from semantic_digital_twin.exceptions import (
    PoseCovarianceNotSixBySixError,
    VariableNotInPoseError,
)
from semantic_digital_twin.spatial_types import PoseCovariance, UncertainPose
from semantic_digital_twin.testing import StateChangeCounter
from semantic_digital_twin.world import (
    ResetStateContextManager,
    WorldModelUpdateContextManager,
    WorldStateBatchContextManager,
)
from semantic_digital_twin.world_description.mesh_file_storage import MeshFileStorage

# remove classes that should not be mapped
ignore_classes = {
    ResetStateContextManager,
    WorldModelUpdateContextManager,
    WorldStateBatchContextManager,
    StateChangeCounter,
    ForwardKinematicsManager,
    MeshFileStorage,
    semantic_digital_twin.adapters.procthor.procthor_resolver.ProcthorResolver,
    SemanticDirection,
    SubclassJSONSerializer,
    # How uncertain a reported pose is, and that pose carried together with it, neither
    # of which anything stores in the world: they are read from a live input and used
    # within a control cycle. The errors go with them - each carries a tuple of any
    # length, which has no column type.
    PoseCovariance,
    PoseCovarianceNotSixBySixError,
    VariableNotInPoseError,
    UncertainPose,
    # A symbolic operation is a step of a query, not something a world stores, so none of
    # them is mapped. The modules defining them are imported above so that they are all
    # declared by the time this is read.
    *recursive_subclasses(SymbolicCallable),
}


def generate_orm():
    """
    Generate the ORM classes for the coraplex package.
    """
    logging.basicConfig(level=logging.INFO)  # Or your preferred config
    logging.getLogger("krrood").setLevel(logging.DEBUG)

    ormatic = ORMatic.from_package(
        [semantic_digital_twin],
        ormatic_interface_dependencies=[],
        ignored_classes=ignore_classes,
        type_mappings={
            trimesh.Trimesh: semantic_digital_twin.orm.model.TrimeshType,
        },
    )
    ormatic.make_all_tables()
    ormatic_interface_path = (
        Path(__file__).parent.parent
        / "src"
        / "semantic_digital_twin"
        / "orm"
        / "ormatic_interface.py"
    )

    with open(ormatic_interface_path, "w") as f:
        ormatic.to_sqlalchemy_file(f)


if __name__ == "__main__":
    generate_orm()
