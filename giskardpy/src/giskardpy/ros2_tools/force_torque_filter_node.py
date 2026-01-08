#!/usr/bin/env python3
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Vector3, Wrench
from geometry_msgs.msg import WrenchStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from scipy.signal import butter, sosfilt


@dataclass
class FilterConfig:
    topic_in: str = "/in"
    topic_filtered_out_suffix: str = "filtered"
    topic_diff_out_suffix: str = "filtered/derivative"
    expected_rate_hz: float = 100.0
    cutoff_main_hz: float = 5.0
    order_main: int = 3
    cutoff_diff_hz: float = 3.0
    order_diff: int = 2
    offset_mode: str = "ewma"  # "ewma" or "none"
    offset_alpha: float = 0.005
    warmup_samples: int = 50
    qos_depth: int = 10
    qos_reliable: bool = True

    def __post_init__(self) -> None:
        self.sanity_check_params()

    def sanity_check_params(self):
        if self.offset_mode not in ("ewma", "none"):
            raise ValueError("Parameter 'offset_mode' must be 'ewma' or 'none'")
        if self.order_main <= 0 or self.order_diff <= 0:
            raise ValueError("Filter orders must be positive integers")
        if self.expected_rate_hz <= 0.0:
            raise ValueError("expected_rate_hz must be > 0")

    @property
    def topic_filtered_out(self) -> str:
        return (
            self.topic_in.rstrip("/") + "/" + self.topic_filtered_out_suffix.lstrip("/")
        )

    @property
    def topic_diff_out(self) -> str:
        return self.topic_in.rstrip("/") + "/" + self.topic_diff_out_suffix.lstrip("/")

    @classmethod
    def from_ros2_params(cls, node: Node) -> FilterConfig:
        cls._declare_parameters_from_cfg(node)
        return cls._read_parameters_to_cfg(node)

    @classmethod
    def _declare_parameters_from_cfg(cls, node: Node) -> None:
        for f in dataclasses.fields(cls):
            # Compute a neutral default for required fields (no default provided)
            if f.default is not dataclasses.MISSING:
                default_val = f.default
            elif getattr(f, "default_factory", dataclasses.MISSING) is not dataclasses.MISSING:  # type: ignore[attr-defined]
                default_val = f.default_factory()  # type: ignore[misc]
            else:
                # Required without default: declare with neutral type-based default
                if f.type is str:
                    default_val = ""
                elif f.type is bool:
                    default_val = False
                elif f.type is int:
                    default_val = 0
                else:
                    # fallback float/others
                    default_val = 0.0
            node.declare_parameter(f.name, default_val)

    @classmethod
    def _read_parameters_to_cfg(cls, node: Node) -> FilterConfig:
        # Typed getters for maximum ROS 2 compatibility across distros
        def get_str(name: str) -> str:
            return str(node.get_parameter(name).get_parameter_value().string_value)

        def get_bool(name: str) -> bool:
            return bool(node.get_parameter(name).get_parameter_value().bool_value)

        def get_int(name: str) -> int:
            return int(node.get_parameter(name).get_parameter_value().integer_value)

        def get_float(name: str) -> float:
            return float(node.get_parameter(name).get_parameter_value().double_value)

        kwargs = {}
        for f in dataclasses.fields(cls):
            if f.type == "str":
                kwargs[f.name] = get_str(f.name)
            elif f.type == "bool":
                kwargs[f.name] = get_bool(f.name)
            elif f.type == "int":
                kwargs[f.name] = get_int(f.name)
            else:
                # Treat anything else as float in this config
                kwargs[f.name] = get_float(f.name)

        return cls(**kwargs)


@dataclass
class OffsetEstimator:
    """
    Estimates and removes a constant or slowly varying offset via EWMA.
    """

    alpha: float
    value: float = 0.0
    initialized: bool = False

    def update(self, x: float) -> float:
        """Update offset with new sample and return the offset estimate."""
        if not self.initialized:
            self.value = float(x)
            self.initialized = True
            return self.value
        self.value = (1.0 - float(self.alpha)) * self.value + float(self.alpha) * float(
            x
        )
        return self.value


@dataclass
class LowPassButter:
    """
    Stateful Butterworth low-pass filter using SOS form for stability.
    """

    cutoff_hz: float
    fs_hz: float
    order: int
    sos: np.ndarray = field(init=False, repr=False)
    zi: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        nyq = 0.5 * float(self.fs_hz)
        wn = float(self.cutoff_hz) / nyq
        self.sos = butter(int(self.order), wn, btype="low", output="sos")
        # zi per SOS section: shape (n_sections, 2)
        self.zi = np.zeros((self.sos.shape[0], 2))

    def step(self, x: float) -> float:
        """Process a single sample through the filter."""
        y, self.zi = sosfilt(self.sos, [x], zi=self.zi)
        return float(y[-1])


@dataclass
class DerivativeEstimator:
    """
    First derivative via sample-to-sample difference using provided dt.
    """

    prev: Optional[float] = None

    def step(self, x: float, dt: float) -> float:
        if dt <= 0.0 or self.prev is None:
            self.prev = float(x)
            return 0.0
        dx = (float(x) - float(self.prev)) / float(dt)
        self.prev = float(x)
        return float(dx)


@dataclass
class WrenchProcessor:
    """
    Composes offset removal, main low-pass, derivative, and derivative smoothing for six axes.
    """

    cfg: FilterConfig
    remove_offset: bool = field(init=False)
    offset: list = field(init=False, repr=False)
    main: list = field(init=False, repr=False)
    diff: list = field(init=False, repr=False)
    diff_smooth: list = field(init=False, repr=False)

    def __post_init__(self) -> None:
        fs = float(self.cfg.expected_rate_hz)
        self.remove_offset = self.cfg.offset_mode != "none"
        self.offset = [
            OffsetEstimator(self.cfg.offset_alpha) if self.remove_offset else None
            for _ in range(6)
        ]
        self.main = [
            LowPassButter(self.cfg.cutoff_main_hz, fs, self.cfg.order_main)
            for _ in range(6)
        ]
        self.diff = [DerivativeEstimator() for _ in range(6)]
        self.diff_smooth = [
            LowPassButter(self.cfg.cutoff_diff_hz, fs, self.cfg.order_diff)
            for _ in range(6)
        ]

    @staticmethod
    def _axes_from_wrench(w: Wrench) -> List[float]:
        return [w.force.x, w.force.y, w.force.z, w.torque.x, w.torque.y, w.torque.z]

    @staticmethod
    def _wrench_from_axes(a: List[float]) -> Wrench:
        f = Vector3(x=float(a[0]), y=float(a[1]), z=float(a[2]))
        t = Vector3(x=float(a[3]), y=float(a[4]), z=float(a[5]))
        return Wrench(force=f, torque=t)

    def process(self, w: Wrench, dt: float) -> Tuple[Wrench, Wrench]:
        axes = self._axes_from_wrench(w)
        filtered = [0.0] * 6
        deriv = [0.0] * 6
        for i, x in enumerate(axes):
            x_in = float(x)
            if self.remove_offset and self.offset[i] is not None:
                x_in = x_in - self.offset[i].update(x_in)
            xf = self.main[i].step(x_in)
            dxf = self.diff[i].step(xf, dt)
            dxf_sm = self.diff_smooth[i].step(dxf)
            filtered[i] = xf
            deriv[i] = dxf_sm
        return self._wrench_from_axes(filtered), self._wrench_from_axes(deriv)


@dataclass(eq=False)
class ForceTorqueFilterNode(Node):
    """
    ROS 2 node: subscribes to a raw WrenchStamped and publishes filtered and derivative topics.

    It publishes two topics:
    - filtered: the denoised signal with offset removed
    - filtered/diff: the first derivative of the filtered signal with reduced noise
    """

    cfg: FilterConfig | None = None
    processor: WrenchProcessor = field(init=False, repr=False)
    last_stamp_ns: Optional[int] = None
    sample_count: int = 0

    def __post_init__(self) -> None:
        super().__init__("force_torque_filter")

        # Declare parameters with defaults, then read actual values
        if self.cfg is None:
            self.cfg = FilterConfig.from_ros2_params(self)

        qos = QoSProfile(
            reliability=(
                ReliabilityPolicy.RELIABLE
                if self.cfg.qos_reliable
                else ReliabilityPolicy.BEST_EFFORT
            ),
            history=HistoryPolicy.KEEP_LAST,
            depth=self.cfg.qos_depth,
        )

        self.processor = WrenchProcessor(self.cfg)

        self.sub = self.create_subscription(
            WrenchStamped, self.cfg.topic_in, self._on_msg, qos
        )
        self.pub_filtered = self.create_publisher(
            WrenchStamped, self.cfg.topic_filtered_out, qos
        )
        self.pub_diff = self.create_publisher(
            WrenchStamped, self.cfg.topic_diff_out, qos
        )

        self.get_logger().info("ForceTorqueFilterNode started")

    def _on_msg(self, msg: WrenchStamped) -> None:
        """Callback that processes an incoming wrench sample.

        - Computes dt from the message header timestamp
        - Processes the signal to produce filtered and derivative outputs
        - Applies a warm-up gate to avoid startup transients
        """
        stamp = msg.header.stamp
        cur_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        dt = 0.0
        if self.last_stamp_ns is not None:
            dt = max(0.0, (cur_ns - self.last_stamp_ns) / 1e9)
        self.last_stamp_ns = cur_ns

        self.sample_count += 1
        dt_use = dt if dt > 0.0 else 1.0 / max(1e-6, self.cfg.expected_rate_hz)

        filt_wrench, diff_wrench = self.processor.process(msg.wrench, dt_use)

        # Warmup gate to avoid startup transients
        # Do not publish for the first `warmup_samples` messages, inclusive of the Nth sample
        # so that exactly publishing `warmup_samples` inputs produces zero outputs.
        if self.sample_count <= self.cfg.warmup_samples:
            return

        out_filtered = WrenchStamped()
        out_filtered.header = msg.header
        out_filtered.wrench = filt_wrench
        self.pub_filtered.publish(out_filtered)

        out_diff = WrenchStamped()
        out_diff.header = msg.header
        out_diff.wrench = diff_wrench
        self.pub_diff.publish(out_diff)


def main():
    rclpy.init()
    node = ForceTorqueFilterNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
