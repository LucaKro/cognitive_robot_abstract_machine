from typing import Optional, List, Dict

import semantic_world.spatial_types.spatial_types as cas
from giskardpy.god_map import god_map
from giskardpy.data_types.data_types import Derivatives
from collections import defaultdict

from giskardpy.motion_statechart.tasks.task import Task
from giskardpy.symbol_manager import symbol_manager


class BaseArmWeightScaling(Task):
    """
    This goals adds weight scaling constraints with the distance between a tip_link and its goal Position as a
    scaling expression. The larger the scaling expression the more is the base movement used toa achieve
    all other constraints instead of arm movements. When the expression decreases this relation changes to favor
    arm movements instead of base movements.
    """

    def __init__(self,
                 root_link: str,
                 tip_link: str,
                 tip_goal: cas.Point3,
                 arm_joints: List[str],
                 base_joints: List[str],
                 gain: float = 100000,
                 name: Optional[str] = None):
        self.root_link = god_map.world.search_for_link_name(root_link, None)
        self.tip_link = god_map.world.search_for_link_name(tip_link, None)
        if name is None:
            name = f'{self.__class__.__name__}/{self.root_link}/{self.tip_link}'
        super().__init__(name=name)

        root_P_tip = god_map.world.compose_fk_expression(self.root_link, self.tip_link).to_position()
        root_P_goal = god_map.world.transform(self.root_link, tip_goal)
        scaling_exp = root_P_goal - root_P_tip

        list_gains = []
        for t in range(god_map.qp_controller.prediction_horizon):
            gains = defaultdict(dict)
            arm_v = None
            for name in arm_joints:
                vs = god_map.world.joints[god_map.world.search_for_joint_name(name)].free_variables
                for v in vs:
                    v_gain = gain * cas.norm(scaling_exp / v.get_upper_limit(Derivatives.velocity))
                    arm_v = v
                    gains[Derivatives.velocity][v] = v_gain
                    gains[Derivatives.acceleration][v] = v_gain
                    gains[Derivatives.jerk][v] = v_gain
            base_v = None
            for name in base_joints:
                vs = god_map.world.joints[god_map.world.search_for_joint_name(name)].free_variables
                for v in vs:
                    v_gain = gain / 100 * cas.save_division(1, cas.norm(scaling_exp / v.get_upper_limit(Derivatives.velocity)))
                    base_v = v
                    gains[Derivatives.velocity][v] = v_gain
                    gains[Derivatives.acceleration][v] = v_gain
                    gains[Derivatives.jerk][v] = v_gain
            list_gains.append(gains)

        god_map.debug_expression_manager.add_debug_expression('base_scaling', gain * cas.save_division(1, cas.norm(scaling_exp / base_v.get_upper_limit(Derivatives.velocity))))
        god_map.debug_expression_manager.add_debug_expression('arm_scaling', gain * cas.norm(scaling_exp / arm_v.get_upper_limit(Derivatives.velocity)))
        god_map.debug_expression_manager.add_debug_expression('norm', cas.norm(scaling_exp))
        god_map.debug_expression_manager.add_debug_expression('division', 1 / cas.norm(scaling_exp))
        self.add_quadratic_weight_gain('baseToArmScaling', gains=list_gains)


class MaxManipulability(Task):
    """
       This goal maximizes the manipulability of the kinematic chain between root_link and tip_link.
       This chain should only include rotational joint and no linear joints i.e. torso lift joints or odometry joints.
       """
    def __init__(self,
                 root_link: str,
                 tip_link: str,
                 gain: float = 0.5,
                 name: Optional[str] = None,
                 m_threshold: float = 0.15):
        self.root_link = god_map.world.search_for_link_name(root_link, None)
        self.tip_link = god_map.world.search_for_link_name(tip_link, None)
        if name is None:
            name = f'{self.__class__.__name__}/{self.root_link}/{self.tip_link}'
        super().__init__(name=name)

        results = god_map.world.compute_split_chain(self.root_link, self.tip_link, True, True, False, False)
        for joint in results[2]:
            if 'joint' in joint and not god_map.world.is_joint_rotational(joint):
                raise Exception('Non rotational joint in kinematic chain of Maximize Manipulability Goal')

        root_P_tip = god_map.world.compose_fk_expression(self.root_link, self.tip_link).to_position()[:3]

        J = cas.jacobian(root_P_tip, root_P_tip.free_symbols())
        JJT = J.dot(J.T)
        m = cas.sqrt(cas.det(JJT))
        list_gains = []
        for t in range(god_map.qp_controller.prediction_horizon):
            gains = defaultdict(dict)
            for symbol in root_P_tip.free_symbols():
                J_dq = cas.total_derivative(J, [symbol], [1])
                product = cas.matrix_inverse(JJT).dot(J_dq).dot(J.T)
                trace = cas.trace(product)
                v = self.get_free_variable(symbol)
                if t < god_map.qp_controller.prediction_horizon - 2:
                    gains[Derivatives.velocity][v] = cas.if_greater(m, m_threshold, 0, trace * m * -gain)
            list_gains.append(gains)
        self.add_linear_weight_gain(name, gains=list_gains)

        god_map.debug_expression_manager.add_debug_expression(f'mIndex{tip_link}', m)

    def get_free_variable(self, symbol):
        for f in god_map.world.free_variables:
            for d in Derivatives:
                if str(god_map.world.free_variables[f].get_symbol(d)) == str(symbol):
                    return god_map.world.free_variables[f]


class MaxManipulabilityAsEq(Task):
    """
       This goal maximizes the manipulability of the kinematic chain between root_link and tip_link.
       This chain should only include rotational joint and no linear joints i.e. torso lift joints or odometry joints.
       """
    def __init__(self,
                 root_link: str,
                 tip_link: str,
                 gain: float = 5,
                 name: Optional[str] = None,
                 m_threshold: float = 0.15):
        self.root_link = god_map.world.search_for_link_name(root_link, None)
        self.tip_link = god_map.world.search_for_link_name(tip_link, None)
        if name is None:
            name = f'{self.__class__.__name__}/{self.root_link}/{self.tip_link}'
        super().__init__(name=name)

        results = god_map.world.compute_split_chain(self.root_link, self.tip_link, True, True, False, False)
        for joint in results[2]:
            if 'joint' in joint and not god_map.world.is_joint_rotational(joint):
                raise Exception('Non rotational joint in kinematic chain of Maximize Manipulability Goal')

        root_P_tip = god_map.world.compose_fk_expression(self.root_link, self.tip_link).to_position()[:3]

        symbols = root_P_tip.free_symbols()

        J = cas.jacobian(root_P_tip, symbols)
        JJT = J.dot(J.T)
        m = cas.sqrt(cas.det(JJT))

        md: Dict[cas.Symbol, cas.Expression] = {}
        for symbol in root_P_tip.free_symbols():
            J_dq = cas.total_derivative(J, [symbol], [1])
            product = cas.matrix_inverse(JJT).dot(J_dq).dot(J.T)
            trace = cas.trace(product)
            md[symbol] = god_map.qp_controller.mpc_dt * gain * trace

        self.add_equality_constraint(reference_velocity=1,
                                     equality_bound=m,
                                     no_diff=True,
                                     # weight=cas.if_greater(m, m_threshold, 0, 1),
                                     weight=1,
                                     task_expression=md,
                                     name=self.name)

        god_map.debug_expression_manager.add_debug_expression(f'mIndex{tip_link}', m, derivatives_to_plot=[0,1])
        god_map.debug_expression_manager.add_debug_expression(f'mIndex{tip_link}_dot',
                                                              cas.jacobian_with_dict([md], symbols).dot(cas.Expression(symbols)),
                                                              derivative=1,
                                                              derivatives_to_plot=[0,1])


    def get_free_variable(self, symbol):
        for f in god_map.world.free_variables:
            for d in Derivatives:
                if str(god_map.world.free_variables[f].get_symbol(d)) == str(symbol):
                    return god_map.world.free_variables[f]

class MaxManipulabilityAsEq2(Task):
    """
       This goal maximizes the manipulability of the kinematic chain between root_link and tip_link.
       This chain should only include rotational joint and no linear joints i.e. torso lift joints or odometry joints.
       """
    def __init__(self,
                 root_link: str,
                 tip_link: str,
                 gain: float = 5,
                 name: Optional[str] = None,
                 m_threshold: float = 0.15):
        self.root_link = god_map.world.search_for_link_name(root_link, None)
        self.tip_link = god_map.world.search_for_link_name(tip_link, None)
        if name is None:
            name = f'{self.__class__.__name__}/{self.root_link}/{self.tip_link}'
        super().__init__(name=name)

        results = god_map.world.compute_split_chain(self.root_link, self.tip_link, True, True, False, False)
        for joint in results[2]:
            if 'joint' in joint and not god_map.world.is_joint_rotational(joint):
                raise Exception('Non rotational joint in kinematic chain of Maximize Manipulability Goal')

        root_P_tip = god_map.world.compose_fk_expression(self.root_link, self.tip_link).to_position()[:3]

        symbols = root_P_tip.free_symbols()

        J = cas.jacobian(root_P_tip, symbols)
        JJT = J.dot(J.T)
        m = cas.sqrt(cas.det(JJT))

        md = cas.manipulability_dot(root_P_tip)

        self.add_equality_constraint(reference_velocity=1,
                                     equality_bound=m,
                                     no_diff=True,
                                     # weight=cas.if_greater(m, m_threshold, 0, 1),
                                     weight=0.0001,
                                     task_expression=md,
                                     name=self.name)

        god_map.debug_expression_manager.add_debug_expression(f'mIndex{tip_link}', m, derivatives_to_plot=[0,1])
        god_map.debug_expression_manager.add_debug_expression(f'mIndex{tip_link}_dot',
                                                              cas.jacobian_with_dict([md], symbols).dot(cas.Expression(symbols)),
                                                              derivative=1,
                                                              derivatives_to_plot=[0,1])

class MaxManipulabilityAsEq3(Task):
    """
       This goal maximizes the manipulability of the kinematic chain between root_link and tip_link.
       This chain should only include rotational joint and no linear joints i.e. torso lift joints or odometry joints.
       """
    def __init__(self,
                 root_link: str,
                 tip_link: str,
                 gain: float = 5,
                 name: Optional[str] = None,
                 m_threshold: float = 0.15):
        self.root_link = god_map.world.search_for_link_name(root_link, None)
        self.tip_link = god_map.world.search_for_link_name(tip_link, None)
        if name is None:
            name = f'{self.__class__.__name__}/{self.root_link}/{self.tip_link}'
        super().__init__(name=name)

        # results = god_map.world.compute_split_chain(self.root_link, self.tip_link, True, True, False, False)
        # for joint in results[2]:
        #     if 'joint' in joint and not god_map.world.is_joint_rotational(joint):
        #         raise Exception('Non rotational joint in kinematic chain of Maximize Manipulability Goal')

        root_P_tip = god_map.world.compose_fk_expression(self.root_link, self.tip_link).to_position()[:3]

        r_R_c = god_map.world.compose_fk_expression(self.root_link, self.tip_link).to_rotation()
        c_R_r_eval = god_map.world.compose_fk_evaluated_expression(self.tip_link, self.root_link).to_rotation()
        hack = cas.RotationMatrix.from_axis_angle(cas.Vector3((0, 0, 1)), 0.0001)
        frame_R_current = r_R_c.dot(hack)  # hack to avoid singularity
        tip_Q_tipCurrent = c_R_r_eval.dot(frame_R_current).to_quaternion()[:3]

        symbols = root_P_tip.free_symbols()
        e = cas.vstack([root_P_tip])
        J = cas.jacobian(e, symbols)
        JJT = J.dot(J.T)
        m = cas.sqrt(cas.det(JJT))

        self.add_position_constraint(reference_velocity=1,
                                     expr_goal=m_threshold,
                                     weight=0.1,
                                     expr_current=m,
                                     name=self.name)

        god_map.debug_expression_manager.add_debug_expression(f'mIndex {tip_link}', m, derivatives_to_plot=[0,1])
        god_map.debug_expression_manager.add_debug_expression(f'mIndex {tip_link} threshold', m_threshold, derivatives_to_plot=[0,1])
        # god_map.debug_expression_manager.add_debug_expression(f'mIndex{tip_link}_dot',
        #                                                       cas.jacobian_with_dict([md], symbols).dot(cas.Expression(symbols)),
        #                                                       derivative=1,
        #                                                       derivatives_to_plot=[0,1])
        self.observation_expression = cas.less_equal(cas.abs(m_threshold - m), 0.0005)


