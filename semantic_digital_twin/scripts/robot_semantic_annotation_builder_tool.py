#!/usr/bin/env python

from __future__ import annotations

import os
import re
import signal
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import rclpy
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QFileDialog,
    QMessageBox,
    QDialog,
    QDialogButtonBox,
    QScrollArea,
    QSplitter,
    QListWidget,
    QListWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QSlider,
    QFormLayout,
    QComboBox,
    QGroupBox,
    QCheckBox,
    QDoubleSpinBox,
    QStackedWidget,
    QTextEdit,
    QFrame,
    QAbstractItemView,
    QHeaderView,
)

try:
    from jinja2 import Template as JinjaTemplate

    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False

from giskardpy.middleware.ros2 import rospy
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
    ShapeSource,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.minimal_robot import MinimalRobot
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import Color
from semantic_digital_twin.world_description.world_entity import Body

def _filename_to_camel_case(path: str) -> str:
    """Convert a file path's base name (without extension) to CamelCase."""
    stem = os.path.splitext(os.path.basename(path))[0]
    parts = re.split(r"[^a-zA-Z0-9]+", stem)
    return "".join(p.capitalize() for p in parts if p) or "MyRobot"


# ---------------------------------------------------------------------------
# Jinja2 template for the generated robot file
# ---------------------------------------------------------------------------

ROBOT_FILE_TEMPLATE = """\
from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Self, List

from semantic_digital_twin.collision_checking.collision_rules import (
    AvoidExternalCollisions,
    AvoidSelfCollisions,
    SelfCollisionMatrixRule,
)
from semantic_digital_twin.datastructures.definitions import (
    GripperState,
    StaticJointState,
)
from semantic_digital_twin.datastructures.field_of_view import FieldOfView
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_part_mixins import (
{%- for mixin in all_mixins %}
    {{ mixin }},
{%- endfor %}
)
from semantic_digital_twin.robots.robot_parts import (
    AbstractRobot,
{%- for base in all_robot_part_bases %}
    {{ base }},
{%- endfor %}
)
from semantic_digital_twin.spatial_types import Quaternion, Vector3
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)

{% for part in parts_in_order %}
{% if part.part_type != 'Robot' %}
@dataclass(eq=False)
class {{ part.class_name }}({{ part.base_classes_str }}):

    def setup_hardware_interfaces(self):
{%- if part.hw_mode == 'all_active' %}
        self._setup_hardware_interfaces_for_active_connections()
{%- else %}
        pass
{%- endif %}

    def setup_joint_states(self) -> List[JointState]:
{%- if part.joint_states %}
        connections = self.active_connections
        joint_states = []
{%- for js in part.joint_states %}
        {{ js.var_name }} = JointState.from_mapping(
            name=PrefixedName("{{ js.name }}", prefix=self.name.name),
            mapping=dict(zip(connections, {{ js.values }})),
            state_type={{ js.state_type }},
        )
        joint_states.append({{ js.var_name }})
{%- endfor %}
        return joint_states
{%- else %}
        return []
{%- endif %}

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "{{ part.root_link }}"
            ),
{%- if part.tip_link %}
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "{{ part.tip_link }}"
            ),
{%- endif %}
{%- if part.tool_frame_link %}
            tool_frame=robot_root._world.get_body_in_branch_by_name(
                robot_root, "{{ part.tool_frame_link }}"
            ),
            front_facing_orientation=Quaternion(),
{%- endif %}
{%- if part.part_type == 'Camera' %}
            forward_facing_axis=Vector3.{{ part.forward_axis }}(),
            field_of_view=FieldOfView(horizontal_angle={{ part.fov_h }}, vertical_angle={{ part.fov_v }}),
            minimal_height={{ part.min_height }},
            maximal_height={{ part.max_height }},
            default_camera={{ part.is_default_camera }},
{%- endif %}
        )

{% endif %}
{% endfor %}

@dataclass(eq=False)
class {{ robot.class_name }}(AbstractRobot, {{ robot.base_classes_str }}):
    \"\"\"
    TODO: Add robot description here.
    \"\"\"

    @classmethod
    def get_ros_file_path(cls) -> str:
        return "{{ robot.ros_file_path }}"

    @classmethod
    def _get_root_body_name(cls) -> str:
        return "{{ robot.root_link }}"

    def _setup_collision_rules(self):
        # TODO: generate an SRDF with the collision matrix tool and adjust the path below
        # srdf_path = os.path.join(
        #     Path(files("semantic_digital_twin")).parent.parent,
        #     "resources", "collision_configs", "your_robot.srdf",
        # )
        # self._world.collision_manager.add_ignore_collision_rule(
        #     SelfCollisionMatrixRule.from_collision_srdf(srdf_path, self._world)
        # )
        self._world.collision_manager.extend_default_rules(
            [
                AvoidExternalCollisions(
                    buffer_zone_distance=0.05, violated_distance=0.0, robot=self
                ),
                AvoidSelfCollisions(
                    buffer_zone_distance=0.03,
                    violated_distance=0.0,
                    robot=self,
                ),
            ]
        )

    def _setup_velocity_limits(self):
        vel_limits = defaultdict(lambda: 1.0)
        self.tighten_dof_velocity_limits_of_1dof_connections(new_limits=vel_limits)
"""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

PART_TYPES = [
    "Robot",
    "MobileBase",
    "Torso",
    "Arm",
    "Neck",
    "EndEffector",
    "Finger",
    "Camera",
]

KINEMATIC_CHAIN_TYPES = {"Torso", "Arm", "Neck", "Finger"}
HAS_TOOL_FRAME = {"EndEffector"}

STATE_TYPE_OPTIONS = [
    "StaticJointState.PARK",
    "StaticJointState.TRANSPORT",
    "GripperState.OPEN",
    "GripperState.CLOSE",
]


@dataclass
class JointStateSpec:
    name: str
    state_type: str
    values: list
    var_name: str = ""

    def __post_init__(self):
        if not self.var_name:
            self.var_name = self.name.replace(" ", "_").replace("-", "_")


@dataclass
class PartNode:
    class_name: str
    part_type: str
    root_link: str = ""
    tip_link: str = ""
    tool_frame_link: str = ""
    root_auto_propagated: bool = False  # True when root was set by parent propagation
    is_thumb: bool = False
    is_default_camera: bool = True
    fov_h: float = 1.047
    fov_v: float = 0.785
    min_height: float = 0.5
    max_height: float = 2.0
    forward_axis: str = "Z"
    hw_mode: str = "all_active"
    joint_states: List[JointStateSpec] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    parent: str = ""
    ros_file_path: str = ""

    @property
    def needs_tip(self) -> bool:
        return self.part_type in KINEMATIC_CHAIN_TYPES

    @property
    def needs_tool_frame(self) -> bool:
        return self.part_type in HAS_TOOL_FRAME


@dataclass
class RobotAnnotatorInterface:
    world: World = field(init=False)
    robot: Optional[MinimalRobot] = field(init=False, default=None)
    parts: Dict[str, PartNode] = field(init=False, default_factory=dict)
    robot_class_name: str = field(init=False, default="MyRobot")
    ros_file_path: str = field(init=False, default="")
    _active_joints: List[ActiveConnection1DOF] = field(init=False, default_factory=list)

    def __post_init__(self):
        self.world = World()
        with self.world.modify_world():
            self.world.add_body(Body(name=PrefixedName("map")))
        self._register_visualization()
        # Pre-create the robot root node so the tree is never empty on first open
        self.parts[self.robot_class_name] = PartNode(
            class_name=self.robot_class_name, part_type="Robot"
        )

    def _register_visualization(self):
        # world.clear() wipes model_change_callbacks and state_change_callbacks,
        # so we must re-create the publisher after each clear.
        VizMarkerPublisher(
            _world=self.world,
            node=rospy.node,
            shape_source=ShapeSource.COLLISION_ONLY,
        ).with_tf_publisher()

    def load_urdf(self, urdf_path: str):
        robot_world = URDFParser.from_file(urdf_path).parse()
        with self.world.modify_world():
            self.world.clear()
            self.world.add_body(map_body := Body(name=PrefixedName("map")))
            self.world.merge_world(
                robot_world,
                FixedConnection(parent=map_body, child=robot_world.root),
            )
        # world.clear() removed the VizMarkerPublisher from all callback lists;
        # recreate it so the loaded robot becomes visible.
        self._register_visualization()
        self.robot = MinimalRobot.from_world(self.world)
        self._active_joints = [
            c
            for c in self.world.connections
            if isinstance(c, ActiveConnection1DOF)
        ]
        self.rename_robot(_filename_to_camel_case(urdf_path))

    def rename_robot(self, new_name: str):
        if not new_name or new_name == self.robot_class_name:
            return
        old_name = self.robot_class_name
        if old_name in self.parts:
            node = self.parts.pop(old_name)
            node.class_name = new_name
            self.parts[new_name] = node
            for child in node.children:
                if child in self.parts:
                    self.parts[child].parent = new_name
        self.robot_class_name = new_name

    def _generate_default_name(self, part_type: str) -> str:
        return f"{self.robot_class_name}{part_type}"

    def propagate_endpoint_to_children(self, class_name: str):
        """Propagate parent's endpoint (tip for kinematic chains, root otherwise) to direct
        children whose root is still empty or was itself auto-propagated."""
        node = self.parts.get(class_name)
        if node is None:
            return
        endpoint = self._parent_endpoint(node)
        if not endpoint:
            return
        for child_name in node.children:
            child = self.parts.get(child_name)
            if child and (not child.root_link or child.root_auto_propagated):
                child.root_link = endpoint
                child.root_auto_propagated = True

    def get_bodies_in_part_scope(self, part_class_name: str) -> List[str]:
        """All bodies in the selected part's own kinematic branch (root inclusive)."""
        if not self.robot:
            return self.body_names
        node = self.parts.get(part_class_name)
        if node and node.root_link:
            try:
                root_body = self.world.get_body_by_name(node.root_link)
                branch = self.world.get_kinematic_structure_entities_of_branch(root_body)
                return [b.name.name for b in branch]
            except Exception:
                pass
        return self.body_names

    def rename_part(self, old_name: str, new_name: str) -> bool:
        if not new_name or old_name == new_name:
            return False
        if new_name in self.parts:
            return False
        node = self.parts.pop(old_name)
        node.class_name = new_name
        self.parts[new_name] = node
        if node.parent and node.parent in self.parts:
            parent = self.parts[node.parent]
            idx = parent.children.index(old_name)
            parent.children[idx] = new_name
        for child_name in node.children:
            if child_name in self.parts:
                self.parts[child_name].parent = new_name
        if self.robot_class_name == old_name:
            self.robot_class_name = new_name
        return True

    def get_valid_root_bodies(self, part_class_name: str) -> List[str]:
        """Root candidates = descendants of the parent part's root body (or all bodies)."""
        if not self.robot:
            return self.body_names
        node = self.parts.get(part_class_name)
        parent_node = self.parts.get(node.parent) if (node and node.parent) else None
        if parent_node and parent_node.root_link:
            try:
                parent_root = self.world.get_body_by_name(parent_node.root_link)
                branch = self.world.get_kinematic_structure_entities_of_branch(parent_root)
                return [b.name.name for b in branch]
            except Exception:
                pass
        return self.body_names

    def get_valid_tip_bodies(self, part_class_name: str) -> List[str]:
        """Tip candidates = descendants of the part's own root body (excluding root itself)."""
        if not self.robot:
            return self.body_names
        node = self.parts.get(part_class_name)
        if node and node.root_link:
            try:
                root_body = self.world.get_body_by_name(node.root_link)
                branch = self.world.get_kinematic_structure_entities_of_branch(root_body)
                return [b.name.name for b in branch if b.name.name != node.root_link]
            except Exception:
                pass
        return self.body_names

    @property
    def body_names(self) -> List[str]:
        return [b.name.name for b in self.world.bodies_topologically_sorted]

    def highlight_body(self, body_name: str):
        with self.world.modify_world():
            for body in self.world.bodies_with_collision:
                body.collision.dye_shapes(Color(0.8, 0.8, 0.8, 0.5))
            try:
                target = self.world.get_body_by_name(body_name)
                if target in self.world.bodies_with_collision:
                    target.collision.dye_shapes(Color(0.2, 1.0, 0.2, 1.0))
            except KeyError:
                pass

    def set_joint_position(self, joint: ActiveConnection1DOF, value: float):
        with self.world.modify_world():
            joint.position = value

    def add_part(self, class_name: str, part_type: str, parent_class_name: str = ""):
        node = PartNode(class_name=class_name, part_type=part_type)
        node.parent = parent_class_name
        if parent_class_name and parent_class_name in self.parts:
            parent_node = self.parts[parent_class_name]
            endpoint = self._parent_endpoint(parent_node)
            if endpoint:
                node.root_link = endpoint
                node.root_auto_propagated = True
        self.parts[class_name] = node
        if parent_class_name and parent_class_name in self.parts:
            self.parts[parent_class_name].children.append(class_name)

    @staticmethod
    def _parent_endpoint(parent_node: PartNode) -> str:
        """The link that a new child should use as its root."""
        if parent_node.needs_tip:
            return parent_node.tip_link
        return parent_node.root_link

    def remove_part(self, class_name: str):
        node = self.parts.get(class_name)
        if node is None:
            return
        for child in list(node.children):
            self.remove_part(child)
        if node.parent and node.parent in self.parts:
            self.parts[node.parent].children.remove(class_name)
        del self.parts[class_name]

    def reparent(self, class_name: str, new_parent: str):
        node = self.parts.get(class_name)
        if node is None:
            return
        if node.parent and node.parent in self.parts:
            self.parts[node.parent].children.remove(class_name)
        node.parent = new_parent
        if new_parent and new_parent in self.parts:
            self.parts[new_parent].children.append(class_name)

    def topological_order(self, root_class: str) -> List[str]:
        result = []

        def dfs(name: str):
            node = self.parts.get(name)
            if node is None:
                return
            for child in node.children:
                dfs(child)
            result.append(name)

        dfs(root_class)
        return result

    def get_arm_end_effector(self, arm_class: str) -> Optional[str]:
        node = self.parts.get(arm_class)
        if node is None:
            return None
        for child in node.children:
            if self.parts[child].part_type == "EndEffector":
                return child
        return None

    def get_neck_camera(self, neck_class: str) -> Optional[str]:
        node = self.parts.get(neck_class)
        if node is None:
            return None
        for child in node.children:
            if self.parts[child].part_type == "Camera":
                return child
        return None

    def get_base_classes(self, class_name: str) -> List[str]:
        node = self.parts.get(class_name)
        if node is None:
            return []
        pt = node.part_type

        if pt == "Robot":
            return self._infer_robot_mixins(node)
        if pt == "Arm":
            ee = self.get_arm_end_effector(class_name)
            ee_str = ee if ee else "EndEffector"
            return [f"Arm[{ee_str}]"]
        if pt == "Neck":
            cam = self.get_neck_camera(class_name)
            cam_str = cam if cam else "Camera"
            return [f"Neck[{cam_str}]"]
        if pt == "Torso":
            return ["Torso"] + self._infer_torso_arm_mixins(node)
        if pt == "EndEffector":
            return ["EndEffector"] + self._infer_finger_mixins(node)
        if pt == "MobileBase":
            return ["MobileBase"]
        if pt == "Finger":
            return ["Finger"]
        if pt == "Camera":
            return ["Camera"]
        return [pt]

    def _infer_robot_mixins(self, node: PartNode) -> List[str]:
        mixins = []
        arm_children = [c for c in node.children if self.parts[c].part_type == "Arm"]
        torso_children = [c for c in node.children if self.parts[c].part_type == "Torso"]
        mobile_base_children = [c for c in node.children if self.parts[c].part_type == "MobileBase"]
        sensor_children = [c for c in node.children if self.parts[c].part_type == "Camera"]

        if mobile_base_children:
            mixins.append(f"HasMobileBase[{mobile_base_children[0]}]")
        if torso_children:
            mixins.append(f"HasTorso[{torso_children[0]}]")
        mixins += self._arm_mixins(arm_children)
        if sensor_children:
            sensor_str = ", ".join(sensor_children)
            mixins.append(f"HasSensors[{sensor_str}]")
        return mixins

    def _infer_torso_arm_mixins(self, node: PartNode) -> List[str]:
        mixins = []
        arm_children = [c for c in node.children if self.parts[c].part_type == "Arm"]
        neck_children = [c for c in node.children if self.parts[c].part_type == "Neck"]
        sensor_children = [c for c in node.children if self.parts[c].part_type == "Camera"]

        if neck_children:
            mixins.append(f"HasNeck[{neck_children[0]}]")
        mixins += self._arm_mixins(arm_children)
        if sensor_children:
            sensor_str = ", ".join(sensor_children)
            mixins.append(f"HasSensors[{sensor_str}]")
        return mixins

    def _arm_mixins(self, arm_children: List[str]) -> List[str]:
        if len(arm_children) == 0:
            return []
        if len(arm_children) == 1:
            return [f"HasOneArm[{arm_children[0]}]"]
        if len(arm_children) == 2:
            return [f"HasLeftRightArm[{arm_children[0]}, {arm_children[1]}]"]
        arms_str = ", ".join(arm_children)
        return [f"HasArms[{arms_str}]"]

    def _infer_finger_mixins(self, node: PartNode) -> List[str]:
        finger_children = [c for c in node.children if self.parts[c].part_type == "Finger"]
        if len(finger_children) == 0:
            return []
        if len(finger_children) == 2:
            return [f"HasTwoFingers[{finger_children[0]}, {finger_children[1]}]"]
        thumbs = [c for c in finger_children if self.parts[c].is_thumb]
        others = [c for c in finger_children if not self.parts[c].is_thumb]
        if thumbs:
            all_fingers = thumbs + others
        else:
            all_fingers = finger_children
        fingers_str = ", ".join(all_fingers)
        return [f"HasFingers[{fingers_str}]"]

    def _collect_all_mixins(self, parts_in_order: List[PartNode]) -> List[str]:
        mixin_names = set()
        for part in parts_in_order:
            for bc in part.base_classes_str.split(", "):
                bc = bc.strip()
                for mixin in [
                    "HasMobileBase", "HasTorso", "HasOneArm", "HasLeftRightArm",
                    "HasArms", "HasNeck", "HasSensors", "HasFingers", "HasTwoFingers",
                    "HasEndEffector",
                ]:
                    if bc.startswith(mixin):
                        mixin_names.add(mixin)
        return sorted(mixin_names)

    def _collect_all_robot_part_bases(self, parts_in_order: List[PartNode]) -> List[str]:
        base_names = set()
        for part in parts_in_order:
            pt = part.part_type
            if pt in ("MobileBase", "Torso", "Arm", "Neck", "EndEffector", "Finger", "Camera"):
                base_names.add(pt)
        return sorted(base_names)

    def generate_code(self, robot_class_name: str) -> str:
        if not HAS_JINJA2:
            return "# jinja2 is not installed. Run: pip install jinja2"

        robot_node = self.parts.get(robot_class_name)
        if robot_node is None:
            return f"# Error: Robot class '{robot_class_name}' not found in parts."

        ordered_names = self.topological_order(robot_class_name)
        ordered_parts: List[PartNode] = []
        for name in ordered_names:
            node = self.parts[name]
            bases = self.get_base_classes(name)
            node_copy = PartNode(
                class_name=node.class_name,
                part_type=node.part_type,
                root_link=node.root_link,
                tip_link=node.tip_link,
                tool_frame_link=node.tool_frame_link,
                is_thumb=node.is_thumb,
                is_default_camera=node.is_default_camera,
                fov_h=node.fov_h,
                fov_v=node.fov_v,
                min_height=node.min_height,
                max_height=node.max_height,
                forward_axis=node.forward_axis,
                hw_mode=node.hw_mode,
                joint_states=node.joint_states,
                children=node.children,
                parent=node.parent,
                ros_file_path=node.ros_file_path,
            )
            node_copy.base_classes_str = ", ".join(bases)
            ordered_parts.append(node_copy)

        non_robot_parts = [p for p in ordered_parts if p.part_type != "Robot"]
        robot_part = next((p for p in ordered_parts if p.part_type == "Robot"), None)
        if robot_part is None:
            return "# Error: no Robot-typed part found."

        all_mixins = self._collect_all_mixins(ordered_parts)
        all_robot_part_bases = self._collect_all_robot_part_bases(non_robot_parts)

        template = JinjaTemplate(ROBOT_FILE_TEMPLATE)
        return template.render(
            parts_in_order=non_robot_parts,
            robot=robot_part,
            all_mixins=all_mixins,
            all_robot_part_bases=all_robot_part_bases,
        )


# ---------------------------------------------------------------------------
# Dialogs
# ---------------------------------------------------------------------------


class BodySelectionDialog(QDialog):
    def __init__(self, bodies: List[str], title: str = "Select Body", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(300, 400)
        layout = QVBoxLayout(self)

        self._list = QListWidget()
        for body in bodies:
            self._list.addItem(body)
        self._list.itemDoubleClicked.connect(lambda _: self.accept())
        layout.addWidget(self._list)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_selected_body(self) -> Optional[str]:
        item = self._list.currentItem()
        return item.text() if item else None


class AddPartDialog(QDialog):
    def __init__(self, interface: RobotAnnotatorInterface, parent=None):
        super().__init__(parent)
        self._interface = interface
        self._user_edited_name = False
        self.setWindowTitle("Add Robot Part")
        self.setMinimumWidth(320)
        layout = QFormLayout(self)

        self.class_name_edit = QLineEdit()
        self.class_name_edit.setPlaceholderText("e.g. MyRobotLeftArm")
        self.class_name_edit.textEdited.connect(self._on_name_edited)
        layout.addRow("Class Name:", self.class_name_edit)

        self.type_combo = QComboBox()
        for pt in PART_TYPES:
            if pt != "Robot":
                self.type_combo.addItem(pt)
        self.type_combo.currentTextChanged.connect(self._on_type_changed)
        layout.addRow("Part Type:", self.type_combo)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

        self._update_default_name()

    def _on_name_edited(self):
        self._user_edited_name = True

    def _on_type_changed(self, _part_type: str):
        if not self._user_edited_name:
            self._update_default_name()

    def _update_default_name(self):
        default = self._interface._generate_default_name(self.type_combo.currentText())
        self.class_name_edit.setText(default)

    def get_class_name(self) -> str:
        return self.class_name_edit.text().strip()

    def get_part_type(self) -> str:
        return self.type_combo.currentText()


class JointStateEditorDialog(QDialog):
    def __init__(self, joint_names: List[str], existing: Optional[JointStateSpec] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Joint State")
        self.setMinimumWidth(400)
        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g. left_arm_park")
        form.addRow("State Name:", self.name_edit)

        self.type_combo = QComboBox()
        for opt in STATE_TYPE_OPTIONS:
            self.type_combo.addItem(opt)
        form.addRow("State Type:", self.type_combo)

        layout.addLayout(form)
        layout.addWidget(QLabel("Joint values:"))

        self._spinboxes: Dict[str, QDoubleSpinBox] = {}
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        inner_layout = QFormLayout(inner)
        for jn in joint_names:
            sb = QDoubleSpinBox()
            sb.setRange(-10.0, 10.0)
            sb.setSingleStep(0.01)
            sb.setDecimals(4)
            self._spinboxes[jn] = sb
            inner_layout.addRow(jn + ":", sb)
        scroll.setWidget(inner)
        scroll.setMinimumHeight(150)
        layout.addWidget(scroll)

        if existing is not None:
            self.name_edit.setText(existing.name)
            idx = self.type_combo.findText(existing.state_type)
            if idx >= 0:
                self.type_combo.setCurrentIndex(idx)
            for i, jn in enumerate(joint_names):
                if i < len(existing.values):
                    self._spinboxes[jn].setValue(existing.values[i])

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_spec(self, joint_names: List[str]) -> JointStateSpec:
        values = [self._spinboxes[jn].value() for jn in joint_names]
        return JointStateSpec(
            name=self.name_edit.text().strip(),
            state_type=self.type_combo.currentText(),
            values=values,
        )


# ---------------------------------------------------------------------------
# Body list panel
# ---------------------------------------------------------------------------


@dataclass
class BodyListPanel(QWidget):
    interface: RobotAnnotatorInterface
    body_selected: object = field(init=False, default=None)

    def __post_init__(self):
        super().__init__()
        self._selected_body: Optional[str] = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>Bodies</b>"))

        self.list_widget = QListWidget()
        self.list_widget.currentTextChanged.connect(self._on_body_selected)
        layout.addWidget(self.list_widget)

        self.status_label = QLabel("Load a URDF to see bodies.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

    def refresh(self, bodies: Optional[List[str]] = None):
        prev = self._selected_body
        self.list_widget.clear()
        names = bodies if bodies is not None else self.interface.body_names
        for name in names:
            self.list_widget.addItem(name)
        if prev and prev in names:
            hits = self.list_widget.findItems(prev, Qt.MatchExactly)
            if hits:
                self.list_widget.setCurrentItem(hits[0])

    def _on_body_selected(self, name: str):
        self._selected_body = name
        if name:
            self.interface.highlight_body(name)
            self.status_label.setText(f"Selected: {name}")

    @property
    def selected_body(self) -> Optional[str]:
        return self._selected_body


# ---------------------------------------------------------------------------
# Part tree panel
# ---------------------------------------------------------------------------


@dataclass
class PartTreePanel(QWidget):
    interface: RobotAnnotatorInterface

    def __post_init__(self):
        super().__init__()
        self._setup_ui()
        self._rebuild_tree()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        header = QHBoxLayout()
        header.addWidget(QLabel("<b>Part Hierarchy</b>"))
        layout.addLayout(header)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Class", "Type"])
        self.tree.header().setSectionResizeMode(QHeaderView.Stretch)
        self.tree.setDragDropMode(QAbstractItemView.InternalMove)
        self.tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tree.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.tree)

        btn_layout = QHBoxLayout()
        self.btn_add = QPushButton("Add Part")
        self.btn_add.clicked.connect(self._on_add)
        self.btn_remove = QPushButton("Remove Part")
        self.btn_remove.clicked.connect(self._on_remove)
        btn_layout.addWidget(self.btn_add)
        btn_layout.addWidget(self.btn_remove)
        layout.addLayout(btn_layout)

    def _on_add(self):
        selected = self.tree.currentItem()
        parent_class = selected.text(0) if selected else self.interface.robot_class_name

        dialog = AddPartDialog(self.interface, self)
        if dialog.exec_() != QDialog.Accepted:
            return
        class_name = dialog.get_class_name()
        part_type = dialog.get_part_type()
        if not class_name:
            QMessageBox.warning(self, "Missing Name", "Please enter a class name.")
            return
        if class_name in self.interface.parts:
            QMessageBox.warning(self, "Duplicate", f"'{class_name}' already exists.")
            return
        try:
            self.interface.add_part(class_name, part_type, parent_class)
            self._rebuild_tree()
            self.part_selection_changed(class_name)
        except Exception as exc:
            import traceback
            QMessageBox.critical(self, "Error adding part", traceback.format_exc())

    def _on_remove(self):
        item = self.tree.currentItem()
        if item is None:
            return
        class_name = item.text(0)
        if class_name == self.interface.robot_class_name:
            QMessageBox.warning(self, "Error", "Cannot remove the robot root.")
            return
        self.interface.remove_part(class_name)
        self._rebuild_tree()

    def _on_selection_changed(self):
        item = self.tree.currentItem()
        if item:
            self.part_selection_changed(item.text(0))

    def part_selection_changed(self, class_name: str):
        pass  # overridden by Application

    def _rebuild_tree(self):
        self.tree.clear()
        root_name = self.interface.robot_class_name
        if root_name not in self.interface.parts:
            return
        root_item = QTreeWidgetItem([root_name, "Robot"])
        self.tree.addTopLevelItem(root_item)
        self._add_children(root_item, root_name)
        self.tree.expandAll()

    def _add_children(self, parent_item: QTreeWidgetItem, class_name: str):
        node = self.interface.parts.get(class_name)
        if node is None:
            return
        for child_name in node.children:
            child_node = self.interface.parts.get(child_name)
            if child_node is None:
                continue
            child_item = QTreeWidgetItem([child_name, child_node.part_type])
            parent_item.addChild(child_item)
            self._add_children(child_item, child_name)

    def get_selected_class(self) -> Optional[str]:
        item = self.tree.currentItem()
        return item.text(0) if item else None


# ---------------------------------------------------------------------------
# Part config panel
# ---------------------------------------------------------------------------


@dataclass
class PartConfigPanel(QWidget):
    interface: RobotAnnotatorInterface
    body_list_panel: BodyListPanel

    def __post_init__(self):
        super().__init__()
        self._current_class: Optional[str] = None
        self._joint_names: List[str] = []
        self._block_auto_save: bool = False
        self.part_renamed_callback = None  # (old_name, new_name) -> None
        self._setup_ui()

    def _setup_ui(self):
        self._outer = QVBoxLayout(self)
        self._outer.addWidget(QLabel("<b>Part Configuration</b>"))

        # --- Class name (always visible, commit on Enter/focus-loss) ---
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Class Name:"))
        self.class_name_edit = QLineEdit()
        self.class_name_edit.setPlaceholderText("ClassName")
        self.class_name_edit.editingFinished.connect(self._on_class_name_committed)
        name_row.addWidget(self.class_name_edit)
        self._outer.addLayout(name_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        self._layout = QVBoxLayout(inner)
        scroll.setWidget(inner)
        self._outer.addWidget(scroll)

        # --- Links group ---
        links_group = QGroupBox("Links")
        links_form = QFormLayout(links_group)

        self.root_edit = QLineEdit()
        self.root_edit.setPlaceholderText("root body name")
        self.btn_root_from_list = QPushButton("From List")
        self.btn_root_from_list.clicked.connect(lambda: self._set_from_list("root"))
        root_row = QHBoxLayout()
        root_row.addWidget(self.root_edit)
        root_row.addWidget(self.btn_root_from_list)
        links_form.addRow("Root Link:", root_row)

        self.tip_edit = QLineEdit()
        self.tip_edit.setPlaceholderText("tip body name (KinematicChain only)")
        self.btn_tip_from_list = QPushButton("From List")
        self.btn_tip_from_list.clicked.connect(lambda: self._set_from_list("tip"))
        tip_row = QHBoxLayout()
        tip_row.addWidget(self.tip_edit)
        tip_row.addWidget(self.btn_tip_from_list)
        links_form.addRow("Tip Link:", tip_row)

        self.tool_frame_edit = QLineEdit()
        self.tool_frame_edit.setPlaceholderText("tool frame body (EndEffector only)")
        self.btn_tool_from_list = QPushButton("From List")
        self.btn_tool_from_list.clicked.connect(lambda: self._set_from_list("tool"))
        tool_row = QHBoxLayout()
        tool_row.addWidget(self.tool_frame_edit)
        tool_row.addWidget(self.btn_tool_from_list)
        links_form.addRow("Tool Frame:", tool_row)

        self._layout.addWidget(links_group)

        # --- Hardware interfaces group ---
        hw_group = QGroupBox("Hardware Interfaces")
        hw_layout = QVBoxLayout(hw_group)
        self.hw_all_active_radio = QCheckBox("All active connections")
        self.hw_all_active_radio.setChecked(True)
        hw_layout.addWidget(self.hw_all_active_radio)
        self._layout.addWidget(hw_group)

        # --- Camera settings ---
        self.camera_group = QGroupBox("Camera Settings")
        cam_form = QFormLayout(self.camera_group)
        self.fov_h_spin = QDoubleSpinBox()
        self.fov_h_spin.setRange(0.01, 6.28)
        self.fov_h_spin.setSingleStep(0.01)
        self.fov_h_spin.setDecimals(5)
        self.fov_h_spin.setValue(1.047)
        cam_form.addRow("FOV Horizontal (rad):", self.fov_h_spin)

        self.fov_v_spin = QDoubleSpinBox()
        self.fov_v_spin.setRange(0.01, 6.28)
        self.fov_v_spin.setSingleStep(0.01)
        self.fov_v_spin.setDecimals(5)
        self.fov_v_spin.setValue(0.785)
        cam_form.addRow("FOV Vertical (rad):", self.fov_v_spin)

        self.min_h_spin = QDoubleSpinBox()
        self.min_h_spin.setRange(0.0, 5.0)
        self.min_h_spin.setSingleStep(0.01)
        self.min_h_spin.setValue(0.5)
        cam_form.addRow("Min Height (m):", self.min_h_spin)

        self.max_h_spin = QDoubleSpinBox()
        self.max_h_spin.setRange(0.0, 5.0)
        self.max_h_spin.setSingleStep(0.01)
        self.max_h_spin.setValue(2.0)
        cam_form.addRow("Max Height (m):", self.max_h_spin)

        self.forward_axis_combo = QComboBox()
        for ax in ("X", "Y", "Z"):
            self.forward_axis_combo.addItem(ax)
        self.forward_axis_combo.setCurrentText("Z")
        cam_form.addRow("Forward Axis:", self.forward_axis_combo)

        self.default_cam_check = QCheckBox("Default Camera")
        self.default_cam_check.setChecked(True)
        cam_form.addRow("", self.default_cam_check)
        self._layout.addWidget(self.camera_group)
        self.camera_group.setVisible(False)

        # --- Finger settings ---
        self.finger_group = QGroupBox("Finger Settings")
        finger_layout = QVBoxLayout(self.finger_group)
        self.is_thumb_check = QCheckBox("This finger is a thumb")
        finger_layout.addWidget(self.is_thumb_check)
        self._layout.addWidget(self.finger_group)
        self.finger_group.setVisible(False)

        # --- Robot settings ---
        self.robot_group = QGroupBox("Robot Settings")
        robot_form = QFormLayout(self.robot_group)
        self.ros_path_edit = QLineEdit()
        self.ros_path_edit.setPlaceholderText("package://pkg/urdf/robot.urdf")
        robot_form.addRow("ROS File Path:", self.ros_path_edit)
        self._layout.addWidget(self.robot_group)
        self.robot_group.setVisible(False)

        # --- Joint states ---
        js_group = QGroupBox("Joint States")
        js_layout = QVBoxLayout(js_group)
        self.js_list = QListWidget()
        js_layout.addWidget(self.js_list)
        js_btn_layout = QHBoxLayout()
        self.btn_add_js = QPushButton("Add")
        self.btn_add_js.clicked.connect(self._on_add_joint_state)
        self.btn_edit_js = QPushButton("Edit")
        self.btn_edit_js.clicked.connect(self._on_edit_joint_state)
        self.btn_remove_js = QPushButton("Remove")
        self.btn_remove_js.clicked.connect(self._on_remove_joint_state)
        js_btn_layout.addWidget(self.btn_add_js)
        js_btn_layout.addWidget(self.btn_edit_js)
        js_btn_layout.addWidget(self.btn_remove_js)
        js_layout.addLayout(js_btn_layout)
        self._layout.addWidget(js_group)

        self._layout.addStretch()

        # Auto-save: write to the model on every field change
        self.root_edit.textChanged.connect(self._auto_save)
        self.tip_edit.textChanged.connect(self._auto_save)
        self.tool_frame_edit.textChanged.connect(self._auto_save)
        self.hw_all_active_radio.stateChanged.connect(self._auto_save)
        self.fov_h_spin.valueChanged.connect(self._auto_save)
        self.fov_v_spin.valueChanged.connect(self._auto_save)
        self.min_h_spin.valueChanged.connect(self._auto_save)
        self.max_h_spin.valueChanged.connect(self._auto_save)
        self.forward_axis_combo.currentTextChanged.connect(self._auto_save)
        self.default_cam_check.stateChanged.connect(self._auto_save)
        self.is_thumb_check.stateChanged.connect(self._auto_save)
        self.ros_path_edit.textChanged.connect(self._auto_save)

    def load_part(self, class_name: str):
        self._block_auto_save = True
        try:
            self._current_class = class_name
            node = self.interface.parts.get(class_name)
            if node is None:
                return

            self.class_name_edit.setText(node.class_name)
            self.root_edit.setText(node.root_link)
            self.tip_edit.setText(node.tip_link)
            self.tool_frame_edit.setText(node.tool_frame_link)

            tip_enabled = node.needs_tip
            self.tip_edit.setEnabled(tip_enabled)
            self.btn_tip_from_list.setEnabled(tip_enabled)

            tool_enabled = node.needs_tool_frame
            self.tool_frame_edit.setEnabled(tool_enabled)
            self.btn_tool_from_list.setEnabled(tool_enabled)

            self.hw_all_active_radio.setChecked(node.hw_mode == "all_active")

            self.camera_group.setVisible(node.part_type == "Camera")
            self.finger_group.setVisible(node.part_type == "Finger")
            self.robot_group.setVisible(node.part_type == "Robot")

            if node.part_type == "Camera":
                self.fov_h_spin.setValue(node.fov_h)
                self.fov_v_spin.setValue(node.fov_v)
                self.min_h_spin.setValue(node.min_height)
                self.max_h_spin.setValue(node.max_height)
                self.forward_axis_combo.setCurrentText(node.forward_axis)
                self.default_cam_check.setChecked(node.is_default_camera)

            if node.part_type == "Finger":
                self.is_thumb_check.setChecked(node.is_thumb)

            if node.part_type == "Robot":
                self.ros_path_edit.setText(node.ros_file_path)

            self._refresh_joint_states(node)
            self._update_joint_names_for_part(node)
        finally:
            self._block_auto_save = False

    def _update_joint_names_for_part(self, node: PartNode):
        if node.root_link and node.tip_link and self.interface.robot:
            try:
                root_body = self.interface.world.get_body_by_name(node.root_link)
                tip_body = self.interface.world.get_body_by_name(node.tip_link)
                chain = self.interface.world.compute_chain_of_connections(root_body, tip_body)
                self._joint_names = [
                    c.name.name for c in chain if isinstance(c, ActiveConnection1DOF)
                ]
            except Exception:
                self._joint_names = []
        else:
            self._joint_names = [
                c.name.name
                for c in self.interface.world.connections
                if isinstance(c, ActiveConnection1DOF)
            ]

    def _refresh_joint_states(self, node: PartNode):
        self.js_list.clear()
        for js in node.joint_states:
            self.js_list.addItem(f"{js.name}  [{js.state_type}]")

    def _set_from_list(self, link_field: str):
        body = self.body_list_panel.selected_body
        if not body:
            return
        if link_field == "root":
            self.root_edit.setText(body)
        elif link_field == "tip":
            self.tip_edit.setText(body)
        else:
            self.tool_frame_edit.setText(body)

    def _on_class_name_committed(self):
        if self._block_auto_save or self._current_class is None:
            return
        new_name = self.class_name_edit.text().strip()
        if not new_name or new_name == self._current_class:
            return
        if new_name in self.interface.parts:
            QMessageBox.warning(self, "Duplicate", f"A part named '{new_name}' already exists.")
            self.class_name_edit.setText(self._current_class)
            return
        old_name = self._current_class
        if self.interface.rename_part(old_name, new_name):
            self._current_class = new_name
            if self.part_renamed_callback:
                self.part_renamed_callback(old_name, new_name)

    def _auto_save(self):
        if self._block_auto_save:
            return
        old_root = ""
        if self._current_class:
            node = self.interface.parts.get(self._current_class)
            if node:
                old_root = node.root_link
        self._on_apply()
        if self._current_class:
            self.interface.propagate_endpoint_to_children(self._current_class)
            node = self.interface.parts.get(self._current_class)
            if node and node.root_link != old_root:
                self.body_list_panel.refresh(
                    self.interface.get_bodies_in_part_scope(self._current_class)
                )

    def _on_add_joint_state(self):
        if self._current_class is None:
            return
        node = self.interface.parts.get(self._current_class)
        if node is None:
            return
        self._update_joint_names_for_part(node)
        dialog = JointStateEditorDialog(self._joint_names, parent=self)
        if dialog.exec_() == QDialog.Accepted:
            spec = dialog.get_spec(self._joint_names)
            node.joint_states.append(spec)
            self._refresh_joint_states(node)

    def _on_edit_joint_state(self):
        if self._current_class is None:
            return
        node = self.interface.parts.get(self._current_class)
        if node is None:
            return
        idx = self.js_list.currentRow()
        if idx < 0 or idx >= len(node.joint_states):
            return
        self._update_joint_names_for_part(node)
        dialog = JointStateEditorDialog(self._joint_names, existing=node.joint_states[idx], parent=self)
        if dialog.exec_() == QDialog.Accepted:
            node.joint_states[idx] = dialog.get_spec(self._joint_names)
            self._refresh_joint_states(node)

    def _on_remove_joint_state(self):
        if self._current_class is None:
            return
        node = self.interface.parts.get(self._current_class)
        if node is None:
            return
        idx = self.js_list.currentRow()
        if 0 <= idx < len(node.joint_states):
            node.joint_states.pop(idx)
            self._refresh_joint_states(node)

    def _on_apply(self):
        if self._current_class is None:
            return
        node = self.interface.parts.get(self._current_class)
        if node is None:
            return

        new_root = self.root_edit.text().strip()
        if new_root != node.root_link:
            node.root_auto_propagated = False
        node.root_link = new_root
        node.tip_link = self.tip_edit.text().strip()
        node.tool_frame_link = self.tool_frame_edit.text().strip()
        node.hw_mode = "all_active" if self.hw_all_active_radio.isChecked() else "none"

        if node.part_type == "Camera":
            node.fov_h = self.fov_h_spin.value()
            node.fov_v = self.fov_v_spin.value()
            node.min_height = self.min_h_spin.value()
            node.max_height = self.max_h_spin.value()
            node.forward_axis = self.forward_axis_combo.currentText()
            node.is_default_camera = self.default_cam_check.isChecked()

        if node.part_type == "Finger":
            node.is_thumb = self.is_thumb_check.isChecked()

        if node.part_type == "Robot":
            node.ros_file_path = self.ros_path_edit.text().strip()


# ---------------------------------------------------------------------------
# Joint slider panel
# ---------------------------------------------------------------------------


@dataclass
class JointSliderPanel(QWidget):
    interface: RobotAnnotatorInterface

    def __post_init__(self):
        super().__init__()
        self._sliders: Dict[str, QSlider] = {}
        self._labels: Dict[str, QLabel] = {}
        self._joints: Dict[str, ActiveConnection1DOF] = {}
        self._setup_ui()

    def _setup_ui(self):
        outer = QVBoxLayout(self)
        outer.addWidget(QLabel("<b>Joint Positions</b>"))

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(200)
        self._inner = QWidget()
        self._inner_layout = QVBoxLayout(self._inner)
        scroll.setWidget(self._inner)
        outer.addWidget(scroll)

    def refresh(self):
        while self._inner_layout.count():
            item = self._inner_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._sliders.clear()
        self._labels.clear()
        self._joints.clear()

        for joint in self.interface._active_joints:
            name = joint.name.name
            dof = joint.dof
            lo = dof.limits.lower.position if dof.limits.lower.position is not None else -3.14
            hi = dof.limits.upper.position if dof.limits.upper.position is not None else 3.14

            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)

            label = QLabel(f"{name}: 0.000")
            label.setMinimumWidth(200)
            self._labels[name] = label

            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(1000)
            slider.setValue(
                int((0.0 - lo) / (hi - lo) * 1000) if hi != lo else 0
            )

            lo_val = lo
            hi_val = hi

            def make_callback(jnt, lo_, hi_, lbl, nm):
                def cb(val):
                    pos = lo_ + (hi_ - lo_) * val / 1000.0
                    lbl.setText(f"{nm}: {pos:.3f}")
                    self.interface.set_joint_position(jnt, pos)
                return cb

            slider.valueChanged.connect(make_callback(joint, lo_val, hi_val, label, name))

            row_layout.addWidget(label)
            row_layout.addWidget(slider)
            self._inner_layout.addWidget(row)
            self._sliders[name] = slider
            self._joints[name] = joint

        if not self._sliders:
            self._inner_layout.addWidget(QLabel("No active joints loaded."))


# ---------------------------------------------------------------------------
# Code preview dialog
# ---------------------------------------------------------------------------


class CodePreviewDialog(QDialog):
    def __init__(self, code: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Generated Robot File")
        self.setMinimumSize(700, 600)
        layout = QVBoxLayout(self)

        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(False)
        self.text_edit.setPlainText(code)
        font = self.text_edit.font()
        font.setFamily("Monospace")
        font.setPointSize(10)
        self.text_edit.setFont(font)
        layout.addWidget(self.text_edit)

        btn_layout = QHBoxLayout()
        self.btn_save = QPushButton("Save to File")
        self.btn_save.clicked.connect(self._on_save)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_layout.addWidget(self.btn_save)
        btn_layout.addWidget(btn_close)
        layout.addLayout(btn_layout)

    def _on_save(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Robot File",
            "",
            "Python files (*.py);;All files (*)",
        )
        if path:
            with open(path, "w") as f:
                f.write(self.text_edit.toPlainText())
            QMessageBox.information(self, "Saved", f"Saved to {path}")


# ---------------------------------------------------------------------------
# Main application window
# ---------------------------------------------------------------------------


@dataclass
class Application(QMainWindow):
    interface: RobotAnnotatorInterface = field(init=False)
    timer: QTimer = field(init=False, default_factory=QTimer)

    def __post_init__(self):
        super().__init__()
        self.interface = RobotAnnotatorInterface()
        self.timer.start(1000)
        self.timer.timeout.connect(lambda: None)
        self._init_ui()

    def _init_ui(self):
        self.setWindowTitle("Robot Semantic Annotation Builder")
        self.setMinimumSize(1100, 700)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # Top bar: URDF loading
        urdf_bar = QHBoxLayout()
        self.urdf_path_edit = QLineEdit()
        self.urdf_path_edit.setPlaceholderText("Path to URDF file…")
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._on_browse_urdf)
        self.btn_load_urdf = QPushButton("Load URDF")
        self.btn_load_urdf.clicked.connect(self._on_load_urdf)
        urdf_bar.addWidget(QLabel("URDF:"))
        urdf_bar.addWidget(self.urdf_path_edit)
        urdf_bar.addWidget(btn_browse)
        urdf_bar.addWidget(self.btn_load_urdf)
        main_layout.addLayout(urdf_bar)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(line)

        # Three-panel splitter
        splitter = QSplitter(Qt.Horizontal)

        self.body_list_panel = BodyListPanel(self.interface)
        splitter.addWidget(self.body_list_panel)

        self.part_tree_panel = PartTreePanel(self.interface)
        self.part_tree_panel.part_selection_changed = self._on_part_selected
        splitter.addWidget(self.part_tree_panel)

        self.config_panel = PartConfigPanel(self.interface, self.body_list_panel)
        self.config_panel.part_renamed_callback = self._on_part_renamed
        splitter.addWidget(self.config_panel)

        splitter.setSizes([250, 350, 400])
        main_layout.addWidget(splitter, stretch=1)

        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)
        line2.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(line2)

        # Joint sliders
        self.slider_panel = JointSliderPanel(self.interface)
        main_layout.addWidget(self.slider_panel)

        # Bottom: generate button
        bottom_bar = QHBoxLayout()
        self.btn_generate = QPushButton("Generate Robot File")
        self.btn_generate.setMinimumHeight(36)
        self.btn_generate.clicked.connect(self._on_generate)
        bottom_bar.addStretch()
        bottom_bar.addWidget(self.btn_generate)
        main_layout.addLayout(bottom_bar)

    def _on_browse_urdf(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open URDF",
            "",
            "URDF files (*.urdf *.xacro);;All files (*)",
        )
        if path:
            self.urdf_path_edit.setText(path)

    def _on_load_urdf(self):
        path = self.urdf_path_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "No path", "Please enter a URDF path.")
            return
        try:
            self.interface.load_urdf(path)
            self.body_list_panel.refresh()
            self.slider_panel.refresh()
            self.part_tree_panel._rebuild_tree()
            QMessageBox.information(
                self,
                "Loaded",
                f"Loaded URDF with {len(list(self.interface.world.bodies))} bodies "
                f"and {len(self.interface._active_joints)} active joints.",
            )
        except Exception as e:
            QMessageBox.critical(self, "Error loading URDF", str(e))

    def _on_part_selected(self, class_name: str):
        if class_name in self.interface.parts:
            self.config_panel.load_part(class_name)
            filtered = self.interface.get_bodies_in_part_scope(class_name)
            self.body_list_panel.refresh(filtered)

    def _on_part_renamed(self, _old_name: str, _new_name: str):
        self.part_tree_panel._rebuild_tree()

    def _on_generate(self):
        robot_name = self.interface.robot_class_name
        if robot_name not in self.interface.parts:
            QMessageBox.warning(
                self,
                "No Robot",
                "No robot root found. Use 'Add Part' with type Robot in the Part Hierarchy panel.",
            )
            return
        code = self.interface.generate_code(robot_name)
        dialog = CodePreviewDialog(code, self)
        dialog.exec_()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def handle_sigint(sig, frame):
    rospy.shutdown()
    QApplication.quit()


if __name__ == "__main__":
    rospy.init_node("robot_semantic_annotation_builder")
    signal.signal(signal.SIGINT, handle_sigint)

    app = QApplication(sys.argv)
    window = Application()
    window.show()
    exit_code = app.exec_()
    rospy.shutdown()
    sys.exit(exit_code)
