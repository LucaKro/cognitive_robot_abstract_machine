from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, TYPE_CHECKING, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from giskardpy.middleware import get_middleware
from giskardpy.motion_statechart.context import ExecutionContext
from giskardpy.motion_statechart.data_types import (
    LifeCycleValues,
    ObservationStateValues,
)
from giskardpy.motion_statechart.graph_node import Goal, MotionStatechartNode
from giskardpy.motion_statechart.plotters.styles import (
    LiftCycleStateToColor,
    ObservationStateToColor,
)
from giskardpy.utils.utils import create_path

if TYPE_CHECKING:
    from giskardpy.motion_statechart.motion_statechart import (
        MotionStatechart,
    )


@dataclass
class HistoryGanttChartPlotter:
    """
    Plot a hierarchy-aware Gantt chart of node states.

    Shows parent-child relationships of Goals by ordering rows in
    preorder and by prefixing labels with tree glyphs (├─, └─, │).
    Optional background bands and goal outlines emphasize grouping.
    """

    motion_statechart: MotionStatechart
    second_length_in_cm: float = 2.0

    def plot_gantt_chart(
        self, file_name: str, context: ExecutionContext | None = None
    ) -> None:
        """
        Render the Gantt chart and save it.

        The chart shows life cycle (top half) and observation state (bottom half)
        per node over time. If a context with dt is provided, the x-axis is in seconds; otherwise, control cycles are used.
        """

        nodes = self.motion_statechart.nodes
        if len(nodes) == 0:
            get_middleware().logwarn(
                "Gantt chart skipped: no nodes in motion statechart."
            )
            return

        history = self.motion_statechart.history.history
        if len(history) == 0:
            get_middleware().logwarn("Gantt chart skipped: empty StateHistory.")
            return

        ordered = self._sort_nodes_by_parents()

        seconds_per_cycle = None
        if context is not None:
            seconds_per_cycle = float(context.dt)
        # time span based on number of history items
        total_cycles = self.motion_statechart.history.history[-1].control_cycle
        time_span_seconds = (
            total_cycles * seconds_per_cycle if seconds_per_cycle else None
        )
        num_bars = len(self.motion_statechart.history.history[0].life_cycle_state)
        figure_width, figure_height = self._compute_figure_size(
            num_bars, time_span_seconds, total_cycles
        )

        # store for drawing
        self._seconds_per_cycle = seconds_per_cycle if seconds_per_cycle else 1.0

        plt.figure(figsize=(figure_width, figure_height))
        plt.grid(True, axis="x", zorder=-1)

        self._iterate_history_and_draw(ordered_nodes=ordered)

        self._format_axes(ordered_nodes=ordered)
        self._save_figure(file_name=file_name)

    def _sort_nodes_by_parents(self) -> List[MotionStatechartNode]:

        def return_children_in_order(n: MotionStatechartNode):
            yield n
            if isinstance(n, Goal):
                for c in n.nodes:
                    yield from return_children_in_order(c)

        ordered_: List[MotionStatechartNode] = []
        for root in self.motion_statechart.top_level_nodes:
            ordered_.extend(list(return_children_in_order(root)))
        # reverse list because plt plots bars bottom to top
        return list(reversed(ordered_))

    def _compute_figure_size(
        self, num_bars: int, time_span_seconds: float | None, cycles_span: int | None
    ) -> tuple[float, float]:
        figure_height = 0.7 + num_bars * 0.25
        if time_span_seconds is not None:
            # 1 inch = 2.54 cm; map seconds to figure width via second_length_in_cm
            inches_per_second = self.second_length_in_cm / 2.54
            figure_width = inches_per_second * time_span_seconds
        else:
            # fallback to cycles scaling
            figure_width = 0.5 * float((cycles_span or 0) + 1)
        return figure_width, figure_height

    def _iterate_history_and_draw(
        self,
        ordered_nodes: List[MotionStatechartNode],
    ) -> None:
        for node_idx, node in enumerate(ordered_nodes):
            self._plot_lifecycle_bar(node=node, node_idx=node_idx)
            self._plot_observation_bar(node=node, node_idx=node_idx)

    def _plot_lifecycle_bar(
        self,
        node: MotionStatechartNode,
        node_idx: int,
    ):
        life_cycle_history = (
            self.motion_statechart.history.get_life_cycle_history_of_node(node)
        )
        control_cycle_indices = [
            h.control_cycle for h in self.motion_statechart.history.history
        ]
        self._plot_node_bar(
            node_idx=node_idx,
            history=life_cycle_history,
            control_cycle_indices=control_cycle_indices,
            color_map=LiftCycleStateToColor,
            top=True,
        )

    def _plot_observation_bar(
        self,
        node: MotionStatechartNode,
        node_idx: int,
    ):
        obs_history = self.motion_statechart.history.get_observation_history_of_node(
            node
        )
        control_cycle_indices = [
            h.control_cycle for h in self.motion_statechart.history.history
        ]
        self._plot_node_bar(
            node_idx=node_idx,
            history=obs_history,
            control_cycle_indices=control_cycle_indices,
            color_map=ObservationStateToColor,
            top=False,
        )

    def _plot_node_bar(
        self,
        node_idx: int,
        history: List[LifeCycleValues | ObservationStateValues],
        control_cycle_indices: List[int],
        color_map: Dict[LifeCycleValues | ObservationStateValues, str],
        top: bool,
    ) -> None:
        current_state = history[0]
        start_idx = 0
        for idx, next_state in zip(control_cycle_indices[1:], history[1:]):
            if current_state != next_state:
                life_cycle_width = (idx + 1 - start_idx) * self._seconds_per_cycle
                self._draw_block(
                    node_idx=node_idx,
                    block_start=start_idx * self._seconds_per_cycle,
                    block_width=life_cycle_width,
                    color=color_map[current_state],
                    top=top,
                )
                start_idx = idx + 1
                current_state = next_state
        last_idx = control_cycle_indices[-1]
        life_cycle_width = (last_idx - start_idx) * self._seconds_per_cycle
        self._draw_block(
            node_idx=node_idx,
            block_start=start_idx * self._seconds_per_cycle,
            block_width=life_cycle_width,
            color=color_map[current_state],
            top=top,
        )

    def _draw_block(
        self,
        node_idx,
        block_start,
        block_width,
        color,
        top: bool,
        bar_height: float = 0.8,
    ):
        if top:
            y = node_idx + bar_height / 4
        else:
            y = node_idx - bar_height / 4
        plt.barh(
            y,
            block_width,
            height=bar_height / 2,
            left=block_start,
            color=color,
            zorder=2,
        )

    def _format_axes(
        self,
        ordered_nodes: List[MotionStatechartNode],
    ) -> None:
        total_cycles = self.motion_statechart.history.history[-1].control_cycle
        total_seconds = total_cycles * self._seconds_per_cycle
        if self._seconds_per_cycle != 1.0:
            plt.xlabel("Time [s]")
            plt.xlim(0, total_seconds)
            ticks = np.arange(0.0, total_seconds + 1e-9, 0.5)
            plt.xticks(ticks)
        else:
            plt.xlabel("Control cycle")
            plt.xlim(0, total_cycles)
            plt.xticks(
                np.arange(
                    0,
                    total_cycles + 1,
                    max(1, (total_cycles - 0 + 1) // 10),
                )
            )
        plt.ylabel("Nodes")
        num_bars = len(self.motion_statechart.history.history[0].life_cycle_state)
        plt.ylim(-0.8, num_bars - 1 + 0.8)

        def make_label(node: MotionStatechartNode, prev_depth: int) -> str:
            depth = node.depth
            if depth == 0:
                return node.unique_name
            diff = depth - prev_depth
            if diff > 0:
                return "│  " * (depth - diff) + "└─ " * diff + node.unique_name
            else:
                return "│  " * (depth - 1) + "├─ " + node.unique_name

        node_names = []
        for idx, n in enumerate(ordered_nodes):
            if idx == 0:
                prev_depth = 0
            else:
                prev_depth = ordered_nodes[idx - 1].depth
            node_names.append(make_label(n, prev_depth))
        node_idx = list(range(len(node_names)))
        plt.yticks(node_idx, node_names)
        plt.gca().yaxis.tick_right()
        plt.tight_layout()

    def _save_figure(self, file_name: str) -> None:
        create_path(file_name)
        plt.savefig(file_name)
        plt.close()
        get_middleware().loginfo(f"Saved gantt chart to {file_name}.")
