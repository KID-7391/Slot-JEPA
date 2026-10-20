import numpy as np
import torch

from realm.environments.utils import *
from realm.helpers import compute_rot_diff_magnitude
from realm.robots.droid_joint_controller import IndividualJointPDController as DROIDJointPDController
from realm.robots.droid_gripper_controller import MultiFingerGripperController as DROIDGripperController
from realm.robots.custom_joint_controller import IndividualJointPDController
from realm.robots.droid_ee_controller import DroidEndEffectorController
import omnigibson as og
from omnigibson.object_states.contact_bodies import ContactBodies
from omnigibson.controllers import REGISTERED_CONTROLLERS
from omnigibson.object_states.open_state import _get_relevant_joints
from omnigibson.utils.object_utils import compute_base_aligned_bboxes, compute_bbox_offset
from omnigibson.prims.joint_prim import JointPrim, JointType
from omnigibson.prims.rigid_prim import RigidPrim
from omnigibson.objects.dataset_object import DatasetObject


REGISTERED_CONTROLLERS["IndividualJointPDController"] = IndividualJointPDController
REGISTERED_CONTROLLERS["DroidEndEffectorController"] = DroidEndEffectorController
REGISTERED_CONTROLLERS["CustomJointController"] = DROIDJointPDController
REGISTERED_CONTROLLERS["CustomGripperController"] = DROIDGripperController
INIT_OPENNESS_FRACTION = 1.0 #0.5
TASK_PROGRESS_RUBRICS = load_task_progressions()


class RealmEnvironmentBase:
    def __init__(
        self,
        main_objects,
        target_objects,
        task_type,
        robot,
        mo_cfgs
    ):
        self.main_objects = main_objects
        self.target_objects = target_objects

        self.mo_pos_orig = np.array(mo_cfgs[0]["position"])
        self.mo_rot_orig = np.array(mo_cfgs[0]["orientation"] if "orientation" in mo_cfgs[0] else [0, 0, 0, 1])
        self.mo_bbox_orig = np.array(mo_cfgs[0]["bounding_box"])

        self.task_type = task_type
        self.robot = robot
        self.robot_finger_links = {self.robot._links[link] for link in self.robot.finger_link_names[self.robot.default_arm]}

        self.was_lifted = False
        if task_type in TASK_PROGRESS_RUBRICS:
            self.task_progression = TASK_PROGRESS_RUBRICS[task_type]
        else:
            self.task_progression = None

        self.reset_joints()

        self.success_conditions = {
            "REACH": self.check_reach_condition,
            "GRASP": self.check_grasp_condition,
            "TOUCH": self.check_touch_condition,
            "LIFT_SLIGHT": self.check_lift_slight_condition,
            "LIFT_LARGE": self.check_lift_large_condition,
            "ROTATED": self.check_rotated,
            "PUSH": self.check_push,
            "MOVE_CLOSE": self.check_move_close_condition,
            "PLACE_INTO": self.check_place_condition,
            "PLACE_ONTO": self.check_place_onto_condition,
            "TOUCH_AND_MOVE_JOINT": self.check_touching_and_moved_mo_joint,
            "OPEN_JOINT_SMALL": self.check_opened_mo_joint_small,
            "OPEN_JOINT_LARGE": self.check_opened_mo_joint_large,
            "OPEN_JOINT_FULL": self.check_opened_mo_joint_full,
            "CLOSE_JOINT_SMALL": self.check_closed_mo_joint_small,
            "CLOSE_JOINT_LARGE": self.check_closed_mo_joint_large,
            "CLOSE_JOINT_FULL": self.check_closed_mo_joint_full,
            "MOVE_JOINT_SMALL": self.check_moved_mo_joint_small, # TODO: turn faucet
            "MOVE_JOINT_LARGE": self.check_moved_mo_joint_large,
            "MOVE_JOINT_FULL": self.check_moved_mo_joint_full,
            "TOGGLED_ON": self.check_toggled_on_condition,
            "POURED": self.check_pour # TODO: pouring
        }

    def  reset_joints(self, target_drawer_loc: str = "top"):
        if self.task_type in ["open_drawer", "close_drawer"]:
            cabinet = self.main_objects[0]
            init_state_open = self.task_type == "close_drawer"
            self.mo_joint = get_target_drawer_joint(cabinet, target_drawer_loc=target_drawer_loc)

            self.mo_joint._articulation_view.set_max_efforts(torch.tensor([[1.0e8]], dtype=torch.float32), joint_indices=self.mo_joint.dof_indices)
            self.mo_joint._articulation_view.set_gains(kps=torch.tensor([[0.0]]), joint_indices=self.mo_joint.dof_indices)
            self.mo_joint._articulation_view.set_gains(kds=torch.tensor([[1000.0]]), joint_indices=self.mo_joint.dof_indices)

            openable_joints = get_openable_joints(cabinet)
            reset_states = [-1 for _ in openable_joints]
            target_joint_ind = openable_joints.index(self.mo_joint)
            reset_states[target_joint_ind] = INIT_OPENNESS_FRACTION if init_state_open else -1
            reset_joints(openable_joints, reset_states=reset_states)
            self.joint_range = self.mo_joint.upper_limit - self.mo_joint.lower_limit
            self.init_openness_fraction = (self.mo_joint.get_state()[0][
                                               0] - self.mo_joint.lower_limit) / self.joint_range
            for _ in range(30):
                og.sim.step()
            for j in cabinet.joints.values():
                j: JointPrim
                j.keep_still()
            for _ in range(10):
                og.sim.step()

        else:
            self.mo_joint = None

    # ============================== [STATUS] ==============================
    def get_ee_pose(self):
        ee_link_name = self.robot.eef_link_names[self.robot.default_arm]
        ee_link = self.robot.links[ee_link_name]
        return ee_link.get_position_orientation()

    def check_collisions(self):
        self_collision = False
        env_collision = False

        # Cache adjacent links to ignore self-collisions between connected bodies
        if not hasattr(self, "_robot_adjacent_links"):
            self._robot_adjacent_links = set()
            if hasattr(self.robot, "joints"):
                for joint in self.robot.joints.values():
                    b0 = joint.body0
                    b1 = joint.body1
                    if b0 and b1:
                        self._robot_adjacent_links.add(frozenset((b0, b1)))

        robot_links = list(self.robot.links.values())
        robot_link_paths = set(l.prim_path for l in robot_links)
        robot_prim_path = self.robot.prim_path

        # Objects to ignore for environment collision (manipulation targets)
        # We use prefixes to catch links and geoms belonging to these objects
        ignore_obj_roots = [obj.prim_path for obj in self.main_objects + self.target_objects]

        for link in robot_links:
            # Skip root link (usually touching mount/floor)
            if link.name == self.robot.root_link_name:
                continue

            contacts = link.contact_list()
            for contact in contacts:
                # Filter by impulse if available (ignore resting/negligible contacts)
                if hasattr(contact, "impulse"):
                    impulse_val = contact.impulse
                    # Handle structured array if necessary (based on error message)
                    if impulse_val.dtype.names is not None:
                         impulse_vec = np.array([impulse_val['x'], impulse_val['y'], impulse_val['z']])
                    else:
                         impulse_vec = impulse_val
                    
                    if np.linalg.norm(impulse_vec) < 1e-3:
                        continue
                else:
                    continue

                if contact.body0 == link.prim_path:
                    other_path = contact.body1
                else:
                    other_path = contact.body0

                # Check if other_path belongs to the robot
                is_robot = other_path in robot_link_paths or other_path.startswith(robot_prim_path)

                if is_robot:
                    # Ignore collisions between adjacent links
                    # Only applicable if we have exact link paths; otherwise assume it's a valid self-collision
                    if other_path in robot_link_paths:
                        if frozenset((link.prim_path, other_path)) in self._robot_adjacent_links:
                            continue
                    self_collision = True
                else:
                    # Check if it's an allowed environment contact (belongs to main/target objects)
                    is_ignored = any(other_path.startswith(root) for root in ignore_obj_roots)
                    if not is_ignored:
                        env_collision = True

            if self_collision and env_collision:
                break

        return self_collision, env_collision

    # ============================== [SUCCESS METRICS] ==============================
    def _object_contacting_links(self, candidate_obj, links):
        candidate_root = str(candidate_obj.prim_path)
        touching_links = set()
        for link in links:
            try:
                contacts = link.contact_list()
            except Exception:
                continue
            for contact in contacts:
                body0 = str(getattr(contact, "body0", ""))
                body1 = str(getattr(contact, "body1", ""))
                other_path = body1 if body0 == link.prim_path else body0
                if other_path.startswith(candidate_root):
                    touching_links.add(link)
                    break
        return touching_links

    def _raw_robot_touching_obj(self, candidate_obj):
        return bool(self._object_contacting_links(candidate_obj, list(self.robot.links.values())))

    def _raw_place_near_target(self, mo, target, xy_threshold=0.125, z_tolerance=0.08):
        mo_pos = np.asarray(mo.get_position_orientation()[0], dtype=float)
        target_pos = np.asarray(target.get_position_orientation()[0], dtype=float)
        xy_dist = float(np.linalg.norm(mo_pos[:2] - target_pos[:2]))
        return xy_dist < float(xy_threshold) and mo_pos[2] > target_pos[2] - float(z_tolerance)

    def is_grasping(self, obs, candidate_obj):
        finger_joints = obs[self.robot.name]['proprio'][7:9].cpu().numpy()
        is_either_finger_closing = (0.45 - finger_joints[0] > 1e-3 or 0.45 - finger_joints[1] > 1e-3)
        try:
            is_both_fingers_touching_obj = len(
                candidate_obj.states[ContactBodies].get_value().intersection(self.robot_finger_links)) == 2
        except Exception:
            is_both_fingers_touching_obj = len(
                self._object_contacting_links(candidate_obj, list(self.robot_finger_links))
            ) >= 2
        is_robot_touching_obj = self.is_touching(obs, candidate_obj)

        if is_both_fingers_touching_obj and is_robot_touching_obj and is_either_finger_closing:
            return True
        return False

    def is_touching(self, obs, candidate_obj):
        try:
            return bool(self.robot.states[og.object_states.Touching].get_value(candidate_obj))
        except Exception:
            return self._raw_robot_touching_obj(candidate_obj)

    def recompute_task_progression(self, obs):
        reward = 0.0

        if self.task_progression is not None:
            for stage, is_completed_flag in self.task_progression.items():
                checker_function = self.success_conditions.get(stage)
                if is_completed_flag or checker_function(obs):
                    if not is_completed_flag:
                        self.task_progression[stage] = True
                    reward += 1 / len(self.task_progression.keys())
                else:
                    break
            assert 0.0 <= reward <= 1.0
        return reward

    def check_reach_condition(self, obs):
        mo = self.main_objects[0]

        if self.task_type == "reach":
            return self.check_touch_condition(obs)

        if self.task_progression in ["open_close_drawer"]:
            return self.is_touching(obs, mo)

        pos1 = mo.get_position_orientation()[0]
        finger1 = list(self.robot_finger_links)[0]
        pos_finger1 = finger1.get_position_orientation()[0]
        finger2 = list(self.robot_finger_links)[1]
        pos_finger2 = finger2.get_position_orientation()[0]

        distance_1 = np.linalg.norm(pos1 - pos_finger1)
        distance_2 = np.linalg.norm(pos1 - pos_finger2)

        dist = 0.1
        return distance_1 < dist or distance_2 < dist or self.check_touch_condition(obs)

        # TODO: make the distance computation bbox dependent
        # obj_pos = mo.get_position_orientation()[0]
        # xmin, ymin, zmin = (obj_pos - self.mo_bbox_orig / 2).tolist()
        # xmax, ymax, zmax = (obj_pos + self.mo_bbox_orig / 2).tolist()
        #
        # finger_distances = []
        # for finger in list(self.robot_finger_links):
        #     finger_pos = finger.get_position_orientation()[0]
        #     finger_x, finger_y, finger_z = finger_pos.tolist()
        #
        #     closest_x = max(xmin, min(finger_x, xmax))
        #     closest_y = max(ymin, min(finger_y, ymax))
        #     closest_z = max(zmin, min(finger_z, zmax))
        #
        #     dx = finger_x - closest_x
        #     dy = finger_y - closest_y
        #     dz = finger_z - closest_z
        #     dist = np.sum(np.abs([dx, dy, dz]))
        #     finger_distances.append(dist)
        # print(finger_distances)

    def check_grasp_condition(self, obs):
        return self.is_grasping(obs, self.main_objects[0])

    def check_touch_condition(self, obs):
        return self.is_touching(obs, self.main_objects[0])

    def get_mo_joint_openness_fraction(self):
        assert self.mo_joint is not None
        return (self.mo_joint.get_state()[0][0] - self.mo_joint.lower_limit) / self.joint_range

    def get_mo_joint_delta(self):
        openness_fraction = self.get_mo_joint_openness_fraction()
        delta_openness_fraction = self.init_openness_fraction - openness_fraction
        return delta_openness_fraction

    def check_touching_and_moved_mo_joint(self, obs, threshold=0.025):
        delta_openness_fraction = self.get_mo_joint_delta()
        if self.task_type == "open_drawer":
            return self.check_touch_condition(obs) and delta_openness_fraction < threshold
        elif self.task_type == "close_drawer":
            return self.check_touch_condition(obs) and delta_openness_fraction > threshold
        else:
            raise NotImplementedError()

    def check_opened_mo_joint_small(self, obs):
        return self.get_mo_joint_openness_fraction() > 0.125

    def check_opened_mo_joint_large(self, obs):
        return self.get_mo_joint_openness_fraction() > 0.65

    def check_opened_mo_joint_full(self, obs):
        return self.get_mo_joint_openness_fraction() > 0.95

    def check_closed_mo_joint_small(self, obs):
        return self.get_mo_joint_openness_fraction() < 0.875

    def check_closed_mo_joint_large(self, obs):
        return self.get_mo_joint_openness_fraction() < 0.35

    def check_closed_mo_joint_full(self, obs):
        return self.get_mo_joint_openness_fraction() < 0.05

    def check_moved_mo_joint_small(self, obs):
        return self.check_closed_mo_joint_small(obs) or self.check_opened_mo_joint_small(obs)

    def check_moved_mo_joint_large(self, obs):
        return self.check_closed_mo_joint_large(obs) or self.check_opened_mo_joint_large(obs)

    def check_moved_mo_joint_full(self, obs):
        return self.check_closed_mo_joint_large(obs) or self.check_opened_mo_joint_large(obs)

    # NOTE: switched to checking Z axis rotation only, possible it is still bad but seems to be working well now
    def check_rotated(self, obs, rot_threshold=1.1):
        mo = self.main_objects[0]
        mo_rot_curr = mo.get_position_orientation()[1]

        rot_diff = compute_rot_diff_magnitude(self.mo_rot_orig, mo_rot_curr)

        return abs(rot_diff) > rot_threshold

    def check_lift_and_distance_condition(self, distance_threshold=0.05, lift_threshold=0.01):
        mo = self.main_objects[0]
        mo_pos_curr = mo.get_position_orientation()[0]

        distance = np.linalg.norm(mo_pos_curr - self.mo_pos_orig)

        return mo_pos_curr[2] - self.mo_pos_orig[2] > lift_threshold and distance > distance_threshold

    def check_lift_slight_condition(self, obs):
        return self.check_lift_and_distance_condition()  # lifted at least 1cm and traveled at least 5cm

    def check_lift_large_condition(self, obs):
        return self.check_lift_and_distance_condition(distance_threshold=0.1, lift_threshold=0.075)

    def check_push(self, obs):
        mo = self.main_objects[0]
        push_cond = self.check_lift_and_distance_condition(distance_threshold=0.1, lift_threshold=-0.05)
        is_lifted = self.check_lift_and_distance_condition(distance_threshold=-0.05, lift_threshold=0.05)
        self.was_lifted = is_lifted or self.was_lifted
        is_robot_touching_obj = self.is_touching(obs, mo)
        return push_cond and is_robot_touching_obj and not self.was_lifted

    def check_move_close_condition(self, obs):
        assert len(self.main_objects) == 1
        assert len(self.target_objects) == 1

        mo = self.main_objects[0]
        pos1 = mo.get_position_orientation()[0]

        target = self.target_objects[0]
        pos2 = target.get_position_orientation()[0]

        distance = np.linalg.norm(pos1 - pos2)
        return distance < 0.125 #0.075 #TODO: adjust for size of receiver, this might not always be enough it seems

    def check_place_condition(self, obs):
        mo = self.main_objects[0]
        target = self.target_objects[0]
        try:
            inside_or_on_top = mo.states[og.object_states.OnTop].get_value(target) or mo.states[og.object_states.Inside].get_value(target)
        except Exception:
            inside_or_on_top = self._raw_place_near_target(mo, target)
        return inside_or_on_top and not self.is_grasping(obs, mo)

    def check_place_onto_condition(self, obs):
        mo = self.main_objects[0]
        target = self.target_objects[0]
        try:
            on_top = mo.states[og.object_states.OnTop].get_value(target)
        except Exception:
            on_top = self._raw_place_near_target(mo, target)
        return on_top and not self.is_grasping(obs, mo)

    def check_toggled_on_condition(self, obs):
        mo = self.main_objects[0]
        return mo.states[og.object_states.ToggledOn].get_value()

    def check_pour(self):
        return False
