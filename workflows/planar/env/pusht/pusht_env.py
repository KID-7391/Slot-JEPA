# env import
import gym
import einops
from gym import spaces
from pymunk.space_debug_draw_options import SpaceDebugColor
from pymunk.vec2d import Vec2d
from typing import Tuple, Sequence, Dict, Union, Optional
import pygame
import pymunk
import numpy as np
import shapely.geometry as sg
import shapely.ops as so
import cv2
import skimage.transform as st
import pymunk.pygame_util
import collections
from matplotlib import cm
import torch

# @markdown ### **Environment**
# @markdown Defines a PyMunk-based Push-T environment `PushTEnv`.
# @markdown
# @markdown **Goal**: push the gray T-block into the green area.
# @markdown
# @markdown Adapted from [Implicit Behavior Cloning](https://implicitbc.github.io/)


positive_y_is_up: bool = False
"""Make increasing values of y point upwards.

When True::

    y
    ^
    |      . (3, 3)
    |
    |   . (2, 2)
    |
    +------ > x

When False::

    +------ > x
    |
    |   . (2, 2)
    |
    |      . (3, 3)
    v
    y

"""


def farthest_point_sampling(points: np.ndarray, n_points: int, init_idx: int):
    """
    Naive O(N^2)
    """
    assert n_points >= 1
    chosen_points = [points[init_idx]]
    for _ in range(n_points - 1):
        cpoints = np.array(chosen_points)
        all_dists = np.linalg.norm(points[:, None, :] - cpoints[None, :, :], axis=-1)
        min_dists = all_dists.min(axis=1)
        next_idx = np.argmax(min_dists)
        next_pt = points[next_idx]
        chosen_points.append(next_pt)
    result = np.array(chosen_points)
    return result


class PymunkKeypointManager:
    def __init__(
        self,
        local_keypoint_map: Dict[str, np.ndarray],
        color_map: Optional[Dict[str, np.ndarray]] = None,
    ):
        """
        local_keypoint_map:
            "<attribute_name>": (N,2) floats in object local coordinate
        """
        if color_map is None:
            cmap = cm.get_cmap("tab10")
            color_map = dict()
            for i, key in enumerate(local_keypoint_map.keys()):
                color_map[key] = (np.array(cmap.colors[i]) * 255).astype(np.uint8)

        self.local_keypoint_map = local_keypoint_map
        self.color_map = color_map

    @property
    def kwargs(self):
        return {
            "local_keypoint_map": self.local_keypoint_map,
            "color_map": self.color_map,
        }

    @classmethod
    def create_from_pusht_env(cls, env, n_block_kps=9, n_agent_kps=3, seed=0, **kwargs):
        rng = np.random.default_rng(seed=seed)
        local_keypoint_map = dict()
        for name in ["block", "agent"]:
            self = env
            self.space = pymunk.Space()
            if name == "agent":
                self.agent = obj = self.add_circle((256, 400), 15)
                n_kps = n_agent_kps
            else:
                self.block = obj = self.add_tee((256, 300), 0)
                n_kps = n_block_kps

            self.screen = pygame.Surface((512, 512))
            self.screen.fill(pygame.Color("white"))
            draw_options = DrawOptions(self.screen)
            self.space.debug_draw(draw_options)
            # pygame.display.flip()
            img = np.uint8(pygame.surfarray.array3d(self.screen).transpose(1, 0, 2))
            obj_mask = (img != np.array([255, 255, 255], dtype=np.uint8)).any(axis=-1)

            tf_img_obj = cls.get_tf_img_obj(obj)
            xy_img = np.moveaxis(np.array(np.indices((512, 512))), 0, -1)[:, :, ::-1]
            local_coord_img = tf_img_obj.inverse(xy_img.reshape(-1, 2)).reshape(
                xy_img.shape
            )
            obj_local_coords = local_coord_img[obj_mask]

            # furthest point sampling
            init_idx = rng.choice(len(obj_local_coords))
            obj_local_kps = farthest_point_sampling(obj_local_coords, n_kps, init_idx)
            small_shift = rng.uniform(0, 1, size=obj_local_kps.shape)
            obj_local_kps += small_shift

            local_keypoint_map[name] = obj_local_kps

        return cls(local_keypoint_map=local_keypoint_map, **kwargs)

    @staticmethod
    def get_tf_img(pose: Sequence):
        pos = pose[:2]
        rot = pose[2]
        tf_img_obj = st.AffineTransform(translation=pos, rotation=rot)
        return tf_img_obj

    @classmethod
    def get_tf_img_obj(cls, obj: pymunk.Body):
        pose = tuple(obj.position) + (obj.angle,)
        return cls.get_tf_img(pose)

    def get_keypoints_global(
        self, pose_map: Dict[set, Union[Sequence, pymunk.Body]], is_obj=False
    ):
        kp_map = dict()
        for key, value in pose_map.items():
            if is_obj:
                tf_img_obj = self.get_tf_img_obj(value)
            else:
                tf_img_obj = self.get_tf_img(value)
            kp_local = self.local_keypoint_map[key]
            kp_global = tf_img_obj(kp_local)
            kp_map[key] = kp_global
        return kp_map

    def draw_keypoints(self, img, kps_map, radius=1):
        scale = np.array(img.shape[:2]) / np.array([512, 512])
        for key, value in kps_map.items():
            color = self.color_map[key].tolist()
            coords = (value * scale).astype(np.int32)
            for coord in coords:
                cv2.circle(img, coord, radius=radius, color=color, thickness=-1)
        return img

    def draw_keypoints_pose(self, img, pose_map, is_obj=False, **kwargs):
        kp_map = self.get_keypoints_global(pose_map, is_obj=is_obj)
        return self.draw_keypoints(img, kps_map=kp_map, **kwargs)


class DrawOptions(pymunk.SpaceDebugDrawOptions):
    def __init__(self, surface: pygame.Surface) -> None:
        """Draw a pymunk.Space on a pygame.Surface object.

        Typical usage::

        >>> import pymunk
        >>> surface = pygame.Surface((10,10))
        >>> space = pymunk.Space()
        >>> options = pymunk.pygame_util.DrawOptions(surface)
        >>> space.debug_draw(options)

        You can control the color of a shape by setting shape.color to the color
        you want it drawn in::

        >>> c = pymunk.Circle(None, 10)
        >>> c.color = pygame.Color("pink")

        See pygame_util.demo.py for a full example

        Since pygame uses a coordinate system where y points down (in contrast
        to many other cases), you either have to make the physics simulation
        with Pymunk also behave in that way, or flip everything when you draw.

        The easiest is probably to just make the simulation behave the same
        way as Pygame does. In that way all coordinates used are in the same
        orientation and easy to reason about::

        >>> space = pymunk.Space()
        >>> space.gravity = (0, -1000)
        >>> body = pymunk.Body()
        >>> body.position = (0, 0) # will be positioned in the top left corner
        >>> space.debug_draw(options)

        To flip the drawing its possible to set the module property
        :py:data:`positive_y_is_up` to True. Then the pygame drawing will flip
        the simulation upside down before drawing::

        >>> positive_y_is_up = True
        >>> body = pymunk.Body()
        >>> body.position = (0, 0)
        >>> # Body will be position in bottom left corner

        :Parameters:
                surface : pygame.Surface
                    Surface that the objects will be drawn on
        """
        self.surface = surface
        super(DrawOptions, self).__init__()

    def draw_circle(
        self,
        pos: Vec2d,
        angle: float,
        radius: float,
        outline_color: SpaceDebugColor,
        fill_color: SpaceDebugColor,
    ) -> None:
        p = to_pygame(pos, self.surface)
        outer = np.asarray(fill_color.as_int()[:3], dtype=np.float32)
        inner = np.asarray(light_color(fill_color).as_int()[:3], dtype=np.float32)
        # Smooth radial shading makes the pusher light in the center and dark
        # near its edge without adding a second hard circular boundary.
        radius_int = max(1, round(radius))
        for circle_radius in range(radius_int, 0, -1):
            center_weight = 1.0 - circle_radius / radius_int
            color = outer * (1.0 - center_weight) + inner * center_weight
            pygame.draw.circle(
                self.surface,
                tuple(np.clip(color, 0, 255).astype(np.uint8)),
                p,
                circle_radius,
                0,
            )
        # Preserve the full pusher radius while reserving only its outermost
        # two pixels as background. Against the block's dark edge this remains
        # a small visible gap at physical contact.
        pygame.draw.circle(
            self.surface,
            pygame.Color("White"),
            p,
            round(radius),
            2,
        )
        circle_edge = pos + Vec2d(radius, 0).rotated(angle)
        p2 = to_pygame(circle_edge, self.surface)
        line_r = 2 if radius > 20 else 1
        # pygame.draw.lines(self.surface, outline_color.as_int(), False, [p, p2], line_r)

    def draw_segment(self, a: Vec2d, b: Vec2d, color: SpaceDebugColor) -> None:
        p1 = to_pygame(a, self.surface)
        p2 = to_pygame(b, self.surface)

        pygame.draw.aalines(self.surface, color.as_int(), False, [p1, p2])

    def draw_fat_segment(
        self,
        a: Tuple[float, float],
        b: Tuple[float, float],
        radius: float,
        outline_color: SpaceDebugColor,
        fill_color: SpaceDebugColor,
    ) -> None:
        p1 = to_pygame(a, self.surface)
        p2 = to_pygame(b, self.surface)

        # Render collision segments at half their physical radius. This keeps
        # the simulator boundary unchanged while producing a thin visual line.
        visual_radius = max(0.5, radius * 0.5)
        r = round(max(1, visual_radius * 2))
        pygame.draw.lines(self.surface, fill_color.as_int(), False, [p1, p2], r)
        if r > 2:
            orthog = [abs(p2[1] - p1[1]), abs(p2[0] - p1[0])]
            if orthog[0] == 0 and orthog[1] == 0:
                return
            scale = visual_radius / (orthog[0] * orthog[0] + orthog[1] * orthog[1]) ** 0.5
            orthog[0] = round(orthog[0] * scale)
            orthog[1] = round(orthog[1] * scale)
            points = [
                (p1[0] - orthog[0], p1[1] - orthog[1]),
                (p1[0] + orthog[0], p1[1] + orthog[1]),
                (p2[0] + orthog[0], p2[1] + orthog[1]),
                (p2[0] - orthog[0], p2[1] - orthog[1]),
            ]
            pygame.draw.polygon(self.surface, fill_color.as_int(), points)
            pygame.draw.circle(
                self.surface,
                fill_color.as_int(),
                (round(p1[0]), round(p1[1])),
                round(visual_radius),
            )
            pygame.draw.circle(
                self.surface,
                fill_color.as_int(),
                (round(p2[0]), round(p2[1])),
                round(visual_radius),
            )

    def draw_polygon(
        self,
        verts: Sequence[Tuple[float, float]],
        radius: float,
        outline_color: SpaceDebugColor,
        fill_color: SpaceDebugColor,
    ) -> None:
        ps = [to_pygame(v, self.surface) for v in verts]
        pygame.draw.polygon(self.surface, fill_color.as_int(), ps)
        # Compound objects are outlined once after all constituent polygons
        # have been drawn. Outlining here would introduce a false seam inside
        # the T block where its two convex collision shapes meet.

    def draw_dot(
        self, size: float, pos: Tuple[float, float], color: SpaceDebugColor
    ) -> None:
        p = to_pygame(pos, self.surface)
        pygame.draw.circle(self.surface, color.as_int(), p, round(size), 0)


def get_mouse_pos(surface: pygame.Surface) -> Tuple[int, int]:
    """Get position of the mouse pointer in pymunk coordinates."""
    p = pygame.mouse.get_pos()
    return from_pygame(p, surface)


def to_pygame(p: Tuple[float, float], surface: pygame.Surface) -> Tuple[int, int]:
    """Convenience method to convert pymunk coordinates to pygame surface
    local coordinates.

    Note that in case positive_y_is_up is False, this function won't actually do
    anything except converting the point to integers.
    """
    if positive_y_is_up:
        return round(p[0]), surface.get_height() - round(p[1])
    else:
        return round(p[0]), round(p[1])


def from_pygame(p: Tuple[float, float], surface: pygame.Surface) -> Tuple[int, int]:
    """Convenience method to convert pygame surface local coordinates to
    pymunk coordinates
    """
    return to_pygame(p, surface)


def light_color(color: SpaceDebugColor):
    color = np.minimum(
        1.2 * np.float32([color.r, color.g, color.b, color.a]), np.float32([255])
    )
    color = SpaceDebugColor(r=color[0], g=color[1], b=color[2], a=color[3])
    return color


def pymunk_to_shapely(body, shapes):
    geoms = list()
    for shape in shapes:
        if isinstance(shape, pymunk.shapes.Poly):
            verts = [body.local_to_world(v) for v in shape.get_vertices()]
            verts += [verts[0]]
            geoms.append(sg.Polygon(verts))
        else:
            raise RuntimeError(f"Unsupported shape type {type(shape)}")
    geom = sg.MultiPolygon(geoms)
    return geom


def body_silhouette(body, shapes):
    """Return one drawable silhouette for a compound polygon body."""
    polygons = []
    for shape in shapes:
        if not isinstance(shape, pymunk.shapes.Poly):
            continue
        vertices = [body.local_to_world(v) for v in shape.get_vertices()]
        polygons.append(sg.Polygon(vertices))
    if not polygons:
        return None

    # The two collision polygons of a T merely touch. Floating-point rotation
    # can leave their union as a MultiPolygon, which would draw the shared edge
    # as an internal contour. A sub-pixel morphological close joins them for
    # rendering without changing the physics geometry at all.
    join_epsilon = 0.25
    return so.unary_union(
        [polygon.buffer(join_epsilon, join_style=2) for polygon in polygons]
    ).buffer(-join_epsilon, join_style=2)


def _silhouette_parts(silhouette):
    if silhouette is None or silhouette.is_empty:
        return []
    return [silhouette] if isinstance(silhouette, sg.Polygon) else silhouette.geoms


def draw_inward_shading(surface, body, shapes, color, falloff=7.0, strength=0.7):
    """Keep the edge light and smoothly deepen the color toward the center."""
    silhouette = body_silhouette(body, shapes)
    parts = _silhouette_parts(silhouette)
    if not parts:
        return

    mask_surface = pygame.Surface(surface.get_size())
    mask_surface.fill((0, 0, 0))
    for polygon in parts:
        points = [to_pygame(point, surface) for point in polygon.exterior.coords[:-1]]
        pygame.draw.polygon(mask_surface, (255, 255, 255), points)
    mask = pygame.surfarray.array3d(mask_surface)[:, :, 0].T
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    weight = (1.0 - np.exp(-distance / falloff)) * (mask > 0) * strength

    image = pygame.surfarray.array3d(surface).transpose(1, 0, 2).astype(np.float32)
    glow_color = np.asarray(color[:3], dtype=np.float32)
    image = image * (1.0 - weight[..., None]) + glow_color * weight[..., None]
    pygame.surfarray.blit_array(
        surface, np.clip(image, 0, 255).astype(np.uint8).transpose(1, 0, 2)
    )


def blur_circle_edge(surface, center, radius, sigma=0.9):
    """Apply a small Gaussian blur only around a circle's rendered edge."""
    image = pygame.surfarray.array3d(surface).transpose(1, 0, 2).astype(np.float32)
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    height, width = image.shape[:2]
    yy, xx = np.ogrid[:height, :width]
    edge_distance = np.abs(np.hypot(xx - center[0], yy - center[1]) - radius)
    weight = np.exp(-(edge_distance ** 2) / (2.0 * (2.0 * sigma) ** 2)) * 0.8
    weight[edge_distance > 3.0 * sigma] = 0.0
    image = image * (1.0 - weight[..., None]) + blurred * weight[..., None]
    pygame.surfarray.blit_array(
        surface, np.clip(image, 0, 255).astype(np.uint8).transpose(1, 0, 2)
    )


def draw_body_outline(surface, body, shapes, color, width=2):
    """Draw only the exterior silhouette of a compound polygon body."""
    silhouette = body_silhouette(body, shapes)
    if silhouette is None:
        return

    for polygon in _silhouette_parts(silhouette):
        points = [to_pygame(point, surface) for point in polygon.exterior.coords[:-1]]
        if len(points) < 3:
            continue
        pygame.draw.lines(surface, color, closed=True, points=points, width=width)
        pygame.draw.aalines(surface, color, closed=True, points=points)


class PushTEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}
    reward_range = (0.0, 1.0)

    def __init__(
        self,
        legacy=False,
        block_cog=None,
        damping=None,
        render_action=False,
        render_size=224,
        reset_to_state=None,
        relative=True,
        action_scale=100,
        with_velocity=False,
        with_target=True,
        draw_block_outline=True,
        shape="T",  # shape can be "T" <- the original shape, "I", "L", "Z", "square" and "small_tee"
        color="LightSlateGray",
    ):  
        self.shape = shape
        self.color = color
        self._seed = None
        self.seed()
        self.window_size = ws = 512  # The size of the PyGame window
        self.render_size = render_size
        self.sim_hz = 100
        # Local controller params.
        self.k_p, self.k_v = 100, 20  # PD control.z
        self.control_hz = self.metadata["video.frames_per_second"]
        # legcay set_state for data compatibility
        self.legacy = legacy
        self.relative = relative  # relative action space
        self.action_scale = action_scale

        # agent_pos, block_pos, block_angle
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0], dtype=np.float64),
            high=np.array([ws, ws, ws, ws, np.pi * 2], dtype=np.float64),
            shape=(5,),
            dtype=np.float64,
        )

        # ``step`` receives actions before multiplying by ``action_scale``.
        # Relative actions are signed displacements; declaring the old
        # absolute [0, ws] box caused the planner's bound projection to clamp
        # every negative displacement to zero.  A displacement of +/-ws can
        # reach either canvas edge from any current position. Absolute-mode
        # inputs are likewise expressed in pre-scale units.
        if self.relative:
            action_low = np.full(2, -ws / self.action_scale, dtype=np.float64)
            action_high = np.full(2, ws / self.action_scale, dtype=np.float64)
        else:
            action_low = np.zeros(2, dtype=np.float64)
            action_high = np.full(2, ws / self.action_scale, dtype=np.float64)
        self.action_space = spaces.Box(
            low=action_low,
            high=action_high,
            shape=(2,),
            dtype=np.float64,
        )

        self.block_cog = block_cog
        self.damping = damping
        self.render_action = render_action

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        self.screen = None

        self.space = None
        self.teleop = None
        self.render_buffer = None
        self.latest_action = None

        self.with_velocity = with_velocity
        self.with_target = with_target
        self.draw_block_outline = bool(draw_block_outline)
        self.reset_to_state = reset_to_state
        self.coverage_arr = []

    def reset(self):
        self._setup()
        if self.block_cog is not None:
            self.block.center_of_gravity = self.block_cog
        if self.damping is not None:
            self.space.damping = self.damping

        # use legacy RandomState for compatibility
        state = self.reset_to_state
        if state is None:
            rs = self.random_state
            if self.with_velocity:
                state = np.array(
                    [
                        rs.randint(50, 450),
                        rs.randint(50, 450),
                        rs.randint(100, 400),
                        rs.randint(100, 400),
                        rs.randn() * 2 * np.pi - np.pi,
                        0,  # set random velocity to 0
                        0,  # set random velocity to 0
                    ]
                )
            else:
                state = np.array(
                    [
                        rs.randint(50, 450),
                        rs.randint(50, 450),
                        rs.randint(100, 400),
                        rs.randint(100, 400),
                        rs.randn() * 2 * np.pi - np.pi,
                    ]
                )
        self._set_state(state)

        self.coverage_arr = []
        state = self._get_obs()
        visual = self._render_frame("rgb_array")
        proprio = state[:2]
        if self.with_velocity:
            proprio = np.concatenate((proprio, state[5:]))
        observation = {
            "visual": visual,
            "proprio": proprio
        }
        return observation, state

    def step(self, action):
        dt = 1.0 / self.sim_hz
        self.n_contact_points = 0
        n_steps = self.sim_hz // self.control_hz
        if action is not None:
            action = np.array(action) * self.action_scale
            if self.relative:
                action = self.agent.position + action
            # The pusher is kinematic, so collision walls alone cannot reliably
            # prevent a large controller target from driving it outside the image.
            agent_radius = max(shape.radius for shape in self.agent.shapes)
            action = np.clip(
                np.asarray(action, dtype=np.float64),
                agent_radius,
                self.window_size - agent_radius,
            )
            self.latest_action = action
            for i in range(n_steps):
                # Step PD control.
                # self.agent.velocity = self.k_p * (act - self.agent.position)    # P control works too.
                acceleration = self.k_p * (action - self.agent.position) + self.k_v * (
                    Vec2d(0, 0) - self.agent.velocity
                )
                self.agent.velocity += acceleration * dt

                # Step physics.
                self.space.step(dt)
                self._keep_objects_inside_canvas()

        # compute reward
        goal_body = self._get_goal_pose_body(self.goal_pose)
        goal_geom = pymunk_to_shapely(goal_body, self.block.shapes)
        block_geom = pymunk_to_shapely(self.block, self.block.shapes)

        intersection_area = goal_geom.intersection(block_geom).area
        goal_area = goal_geom.area
        coverage = intersection_area / goal_area
        reward = np.clip(coverage / self.success_threshold, 0, 1)
        done = False  # coverage > self.success_threshold

        self.coverage_arr.append(coverage)

        state = self._get_obs()
        # visual = self._render_frame("rgb_array")
        visual = self._render_frame("rgb_array")
        proprio = state[:2]
        if self.with_velocity:
            proprio = np.concatenate((proprio, state[5:]))
        observation = {
            "visual": visual,
            "proprio": proprio
        }
        # observation = (
        #     einops.rearrange(observation, "H W C -> 1 C H W") / 255.0
        # )  # VCHW, range [0, 1]
        info = self._get_info()
        info["state"] = state
        info["max_coverage"] = max(self.coverage_arr)
        info["final_coverage"] = self.coverage_arr[-1]

        return observation, reward, done, info

    def render(self, mode):
        return self._render_frame(mode)

    def teleop_agent(self):
        TeleopAgent = collections.namedtuple("TeleopAgent", ["act"])

        def act(obs):
            act = None
            mouse_position = pymunk.pygame_util.from_pygame(
                Vec2d(*pygame.mouse.get_pos()), self.screen
            )
            if self.teleop or (mouse_position - self.agent.position).length < 30:
                self.teleop = True
                act = mouse_position
            return act

        return TeleopAgent(act)

    def _get_obs(self):
        if self.with_velocity:
            obs = np.array(
                tuple(self.agent.position)
                + tuple(self.block.position)
                + (self.block.angle % (2 * np.pi),)
                + tuple(self.agent.velocity)
            ).astype(np.float32)
        else:
            obs = np.array(
                tuple(self.agent.position)
                + tuple(self.block.position)
                + (self.block.angle % (2 * np.pi),)
            ).astype(np.float32)
        return obs

    def _get_goal_pose_body(self, pose):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (50, 100))
        body = pymunk.Body(mass, inertia)
        # preserving the legacy assignment order for compatibility
        # the order here doesn't matter somehow, maybe because CoM is aligned with body origin
        body.position = pose[:2].tolist()
        body.angle = pose[2]
        return body

    def _get_info(self):
        n_steps = self.sim_hz // self.control_hz
        n_contact_points_per_step = int(np.ceil(self.n_contact_points / n_steps))
        info = {
            "pos_agent": np.array(self.agent.position),
            "vel_agent": np.array(self.agent.velocity),
            "block_pose": np.array(list(self.block.position) + [self.block.angle]),
            "goal_pose": self.goal_pose,
            "n_contacts": n_contact_points_per_step,
        }
        return info

    def _render_frame(self, mode):
        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((245, 245, 245))
        self.screen = canvas

        draw_options = DrawOptions(canvas)

        # The goal pose remains active for reward/success computation but is
        # intentionally hidden to keep the visual observation uncluttered.
        self.space.debug_draw(draw_options)
        blur_circle_edge(
            canvas,
            self.agent.position,
            next(iter(self.agent.shapes)).radius,
        )
        if self.draw_block_outline:
            draw_body_outline(
                canvas,
                self.block,
                self.block.shapes,
                pygame.Color(65, 90, 120),
            )

        if mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # the clock is already ticked during in step for "human"

        img = np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
        if self.render_size != self.window_size:
            interpolation = (
                cv2.INTER_AREA
                if self.render_size < self.window_size
                else cv2.INTER_CUBIC
            )
            img = cv2.resize(
                img,
                (self.render_size, self.render_size),
                interpolation=interpolation,
            )
        if self.render_action:
            if self.render_action and (self.latest_action is not None):
                action = np.array(self.latest_action)
                coord = (action / 512 * 96).astype(np.int32)
                marker_size = int(8 / 96 * self.render_size)
                thickness = int(1 / 96 * self.render_size)
                cv2.drawMarker(
                    img,
                    coord,
                    color=(255, 0, 0),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=marker_size,
                    thickness=thickness,
                )
        return img

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)
        self.random_state = np.random.RandomState(seed)

    def _handle_collision(self, arbiter, space, data):
        self.n_contact_points += len(arbiter.contact_point_set.points)

    def _set_state(self, state):
        if isinstance(state, np.ndarray):
            state = state.tolist()
        pos_agent = state[:2]
        pos_block = state[2:4]
        rot_block = state[4]
        vel_block = tuple(state[5:]) if self.with_velocity else (0, 0)
        self.agent.velocity = vel_block
        self.agent.position = pos_agent
        # setting angle rotates with respect to center of mass
        # therefore will modify the geometric position
        # if not the same as CoM
        # therefore should be modified first.
        if self.legacy:
            # for compatibility with legacy data
            self.block.position = pos_block
            self.block.angle = rot_block
        else:
            self.block.angle = rot_block
            self.block.position = pos_block

        # Position/angle assignments take effect immediately in Pymunk.  Do
        # not advance physics here: when a recorded state contains non-zero
        # agent velocity, an implicit simulation tick shifts the initial
        # position before the first requested action and makes dataset replay
        # inconsistent with predictor input.
        self.space.reindex_shapes_for_body(self.agent)
        self.space.reindex_shapes_for_body(self.block)
        self._keep_objects_inside_canvas()

    def _keep_body_inside_canvas(self, body):
        """Translate a body minimally so all of its shapes remain in the image."""
        boxes = [shape.cache_bb() for shape in body.shapes]
        if not boxes:
            return
        left = min(box.left for box in boxes)
        right = max(box.right for box in boxes)
        bottom = min(box.bottom for box in boxes)
        top = max(box.top for box in boxes)
        dx = -left if left < 0 else self.window_size - right if right > self.window_size else 0
        dy = -bottom if bottom < 0 else self.window_size - top if top > self.window_size else 0
        if dx == 0 and dy == 0:
            return
        body.position = body.position + (dx, dy)
        velocity_x, velocity_y = body.velocity
        if dx and velocity_x * dx < 0:
            velocity_x = 0
        if dy and velocity_y * dy < 0:
            velocity_y = 0
        body.velocity = velocity_x, velocity_y
        self.space.reindex_shapes_for_body(body)

    def _keep_objects_inside_canvas(self):
        self._keep_body_inside_canvas(self.agent)
        self._keep_body_inside_canvas(self.block)

    def _set_state_local(self, state_local):
        agent_pos_local = state_local[:2]
        block_pose_local = state_local[2:]
        tf_img_obj = st.AffineTransform(
            translation=self.goal_pose[:2], rotation=self.goal_pose[2]
        )
        tf_obj_new = st.AffineTransform(
            translation=block_pose_local[:2], rotation=block_pose_local[2]
        )
        tf_img_new = st.AffineTransform(matrix=tf_img_obj.params @ tf_obj_new.params)
        agent_pos_new = tf_img_new(agent_pos_local)
        new_state = np.array(
            list(agent_pos_new[0])
            + list(tf_img_new.translation)
            + [tf_img_new.rotation]
        )
        self._set_state(new_state)
        return new_state

    def set_task_goal(self, goal):
        goal = np.asarray(goal, dtype=np.float64).copy()
        # The translucent target is rendered with the same geometry as the block.
        # Shift its pose as well so no target pixels are clipped by the canvas.
        if self.space is not None and hasattr(self, "block"):
            goal_body = self._get_goal_pose_body(goal)
            left, bottom, right, top = pymunk_to_shapely(
                goal_body, self.block.shapes
            ).bounds
            goal[0] += -left if left < 0 else self.window_size - right if right > self.window_size else 0
            goal[1] += -bottom if bottom < 0 else self.window_size - top if top > self.window_size else 0
        self.goal_pose = goal

    def _setup(self):
        self.space = pymunk.Space()
        self.space.gravity = 0, 0
        self.space.damping = 0
        self.teleop = False
        self.render_buffer = list()

        # Add walls.
        walls = [
            self._add_segment((5, 506), (5, 5), 2),
            self._add_segment((5, 5), (506, 5), 2),
            self._add_segment((506, 5), (506, 506), 2),
            self._add_segment((5, 506), (506, 506), 2),
        ]
        self.space.add(*walls)

        # Add agent, block, and goal zone.
        self.agent = self.add_circle((256, 400), 15)
        # self.block = self.add_tee((256, 300), 0)
        self.block = self.add_shape(self.shape, (256, 300), 0, color=self.color, scale=30)
        if self.with_target:
            self.goal_color = pygame.Color("LightGreen")
        else:
            self.goal_color = pygame.Color("White")
        self.goal_pose = np.array([256, 256, np.pi / 4])  # x, y, theta (in radians)

        # Add collision handling
        self.collision_handeler = self.space.add_collision_handler(0, 0)
        self.collision_handeler.post_solve = self._handle_collision
        self.n_contact_points = 0

        self.max_score = 50 * 100
        self.success_threshold = 0.95  # 95% coverage.

    def _add_segment(self, a, b, radius):
        shape = pymunk.Segment(self.space.static_body, a, b, radius)
        # Keep the physical boundary visible without making it the strongest
        # contrast feature in the image.
        shape.color = pygame.Color(90, 90, 90)
        return shape

    def add_circle(self, position, radius):
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        body.friction = 1
        shape = pymunk.Circle(body, radius)
        shape.color = pygame.Color("RoyalBlue")
        self.space.add(body, shape)
        return body

    def add_box(self, position, height, width):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (height, width))
        body = pymunk.Body(mass, inertia)
        body.position = position
        shape = pymunk.Poly.create_box(body, (height, width))
        shape.color = pygame.Color("LightSlateGray")
        self.space.add(body, shape)
        return body

    _reference_object_area = 7.0 * 30.0**2
    _object_area_density = 1.0 / _reference_object_area

    def _add_grid_compound(
        self,
        position,
        angle,
        unit_polygons,
        scale,
        color,
        mask,
    ):
        """Create a 30px-grid body at the original PushT area density."""
        scale = float(scale)
        if scale <= 0.0:
            raise ValueError("shape scale must be positive")
        polygons = [
            [(float(x * scale), float(y * scale)) for x, y in vertices]
            for vertices in unit_polygons
        ]
        geometries = [sg.Polygon(vertices) for vertices in polygons]
        areas = np.asarray([geometry.area for geometry in geometries])
        centroids = np.asarray(
            [geometry.centroid.coords[0] for geometry in geometries]
        )
        total_area = float(areas.sum())
        center_of_gravity = (areas[:, None] * centroids).sum(axis=0) / total_area
        part_masses = areas * self._object_area_density
        mass = float(part_masses.sum())
        offset = tuple((-center_of_gravity).tolist())
        moment = sum(
            pymunk.moment_for_poly(
                float(part_mass), vertices=vertices, offset=offset
            )
            for part_mass, vertices in zip(part_masses, polygons)
        )
        body = pymunk.Body(mass, moment)
        shapes = [pymunk.Poly(body, vertices) for vertices in polygons]
        for shape in shapes:
            shape.color = pygame.Color(color)
            shape.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = tuple(center_of_gravity.tolist())
        body.position = position
        body.angle = angle
        body.friction = 1
        self.space.add(body, *shapes)
        body.geometry_scale = float(scale)
        # Compatibility for inspection scripts written before geometries
        # stopped being rescaled to equal area. This now denotes only the
        # fixed grid-cell size, not an equal-area scale.
        body.equal_area_scale = float(scale)
        body.geometric_area = total_area
        body.area_density = self._object_area_density
        return body

    def add_tee(
        self, position, angle, scale=30, color="LightSlateGray",
        mask=pymunk.ShapeFilter.ALL_MASKS(),
    ):
        return self._add_grid_compound(
            position, angle,
            [
                [(-2, 1), (2, 1), (2, 0), (-2, 0)],
                [(-0.5, 1), (-0.5, 4), (0.5, 4), (0.5, 1)],
            ],
            scale, color, mask,
        )

    def add_small_tee(
        self, position, angle, scale=30, color="LightSlateGray",
        mask=pymunk.ShapeFilter.ALL_MASKS(),
    ):
        return self._add_grid_compound(
            position, angle,
            [
                [(-2, 0.5), (2, 0.5), (2, -0.5), (-2, -0.5)],
                [(-0.5, 0.5), (-0.5, 2), (0.5, 2), (0.5, 0.5)],
            ],
            scale, color, mask,
        )

    def add_plus(
        self, position, angle, scale=30, color="LightSlateGray",
        mask=pymunk.ShapeFilter.ALL_MASKS(),
    ):
        return self._add_grid_compound(
            position, angle,
            [
                [(-2, 0.5), (2, 0.5), (2, -0.5), (-2, -0.5)],
                [(-0.5, 0.5), (-0.5, 2), (0.5, 2), (0.5, 0.5)],
                [(-0.5, -0.5), (-0.5, -2), (0.5, -2), (0.5, -0.5)],
            ],
            scale, color, mask,
        )

    def add_L(
        self, position, angle, scale=30, color="LightSlateGray",
        mask=pymunk.ShapeFilter.ALL_MASKS(),
    ):
        return self._add_grid_compound(
            position, angle,
            [
                [(0, 0), (0, 3), (1, 3), (1, 0)],
                [(0, 0), (3, 0), (3, -1), (0, -1)],
            ],
            scale, color, mask,
        )

    def add_Z(
        self, position, angle, scale=30, color="LightSlateGray",
        mask=pymunk.ShapeFilter.ALL_MASKS(),
    ):
        return self._add_grid_compound(
            position, angle,
            [
                [(0, 0), (0, 1), (2, 1), (2, 0)],
                [(-1, 0), (1, 0), (1, -1), (-1, -1)],
            ],
            scale, color, mask,
        )

    def add_square(
        self, position, angle, scale=30, color="LightSlateGray",
        mask=pymunk.ShapeFilter.ALL_MASKS(),
    ):
        return self._add_grid_compound(
            position, angle,
            [[(-1.5, -1.5), (-1.5, 1.5), (1.5, 1.5), (1.5, -1.5)]],
            scale, color, mask,
        )

    def add_I(
        self, position, angle, scale=30, color="LightSlateGray",
        mask=pymunk.ShapeFilter.ALL_MASKS(),
    ):
        return self._add_grid_compound(
            position, angle,
            [[(-0.5, -2.5), (-0.5, 2.5), (0.5, 2.5), (0.5, -2.5)]],
            scale, color, mask,
        )

    def add_shape(self, shape, *args, **kwargs):
        # Dispatch method based on the 'shape' parameter
        if shape == "L":
            return self.add_L(*args, **kwargs)
        elif shape == "T":
            return self.add_tee(*args, **kwargs)
        elif shape == "Z":
            return self.add_Z(*args, **kwargs)
        elif shape == "square":
            return self.add_square(*args, **kwargs)
        elif shape == "I":
            return self.add_I(*args, **kwargs)
        elif shape == "small_tee":
            return self.add_small_tee(*args, **kwargs)
        if shape == "+":
            return self.add_plus(*args, **kwargs)
        else:
            raise ValueError(f"Unknown shape type: {shape}")


class PymunkKeypointManager:
    def __init__(
        self,
        local_keypoint_map: Dict[str, np.ndarray],
        color_map: Optional[Dict[str, np.ndarray]] = None,
    ):
        """
        local_keypoint_map:
            "<attribute_name>": (N,2) floats in object local coordinate
        """
        if color_map is None:
            cmap = cm.get_cmap("tab10")
            color_map = dict()
            for i, key in enumerate(local_keypoint_map.keys()):
                color_map[key] = (np.array(cmap.colors[i]) * 255).astype(np.uint8)

        self.local_keypoint_map = local_keypoint_map
        self.color_map = color_map

    @property
    def kwargs(self):
        return {
            "local_keypoint_map": self.local_keypoint_map,
            "color_map": self.color_map,
        }

    @classmethod
    def create_from_pusht_env(cls, env, n_block_kps=9, n_agent_kps=3, seed=0, **kwargs):
        rng = np.random.default_rng(seed=seed)
        local_keypoint_map = dict()
        for name in ["block", "agent"]:
            self = env
            self.space = pymunk.Space()
            if name == "agent":
                self.agent = obj = self.add_circle((256, 400), 15)
                n_kps = n_agent_kps
            else:
                self.block = obj = self.add_tee((256, 300), 0)
                n_kps = n_block_kps

            self.screen = pygame.Surface((512, 512))
            self.screen.fill(pygame.Color("white"))
            draw_options = DrawOptions(self.screen)
            self.space.debug_draw(draw_options)
            # pygame.display.flip()
            img = np.uint8(pygame.surfarray.array3d(self.screen).transpose(1, 0, 2))
            obj_mask = (img != np.array([255, 255, 255], dtype=np.uint8)).any(axis=-1)

            tf_img_obj = cls.get_tf_img_obj(obj)
            xy_img = np.moveaxis(np.array(np.indices((512, 512))), 0, -1)[:, :, ::-1]
            local_coord_img = tf_img_obj.inverse(xy_img.reshape(-1, 2)).reshape(
                xy_img.shape
            )
            obj_local_coords = local_coord_img[obj_mask]

            # furthest point sampling
            init_idx = rng.choice(len(obj_local_coords))
            obj_local_kps = farthest_point_sampling(obj_local_coords, n_kps, init_idx)
            small_shift = rng.uniform(0, 1, size=obj_local_kps.shape)
            obj_local_kps += small_shift

            local_keypoint_map[name] = obj_local_kps

        return cls(local_keypoint_map=local_keypoint_map, **kwargs)

    @staticmethod
    def get_tf_img(pose: Sequence):
        pos = pose[:2]
        rot = pose[2]
        tf_img_obj = st.AffineTransform(translation=pos, rotation=rot)
        return tf_img_obj

    @classmethod
    def get_tf_img_obj(cls, obj: pymunk.Body):
        pose = tuple(obj.position) + (obj.angle,)
        return cls.get_tf_img(pose)

    def get_keypoints_global(
        self, pose_map: Dict[set, Union[Sequence, pymunk.Body]], is_obj=False
    ):
        kp_map = dict()
        for key, value in pose_map.items():
            if is_obj:
                tf_img_obj = self.get_tf_img_obj(value)
            else:
                tf_img_obj = self.get_tf_img(value)
            kp_local = self.local_keypoint_map[key]
            kp_global = tf_img_obj(kp_local)
            kp_map[key] = kp_global
        return kp_map

    def draw_keypoints(self, img, kps_map, radius=1):
        scale = np.array(img.shape[:2]) / np.array([512, 512])
        for key, value in kps_map.items():
            color = self.color_map[key].tolist()
            coords = (value * scale).astype(np.int32)
            for coord in coords:
                cv2.circle(img, coord, radius=radius, color=color, thickness=-1)
        return img

    def draw_keypoints_pose(self, img, pose_map, is_obj=False, **kwargs):
        kp_map = self.get_keypoints_global(pose_map, is_obj=is_obj)
        return self.draw_keypoints(img, kps_map=kp_map, **kwargs)


class PushTKeypointsEnv(PushTEnv):
    def __init__(
        self,
        legacy=False,
        block_cog=None,
        damping=None,
        render_size=96,
        keypoint_visible_rate=1.0,
        agent_keypoints=False,
        draw_keypoints=False,
        reset_to_state=None,
        render_action=False,
        local_keypoint_map: Dict[str, np.ndarray] = None,
        color_map: Optional[Dict[str, np.ndarray]] = None,
    ):
        super().__init__(
            legacy=legacy,
            block_cog=block_cog,
            damping=damping,
            render_size=render_size,
            reset_to_state=reset_to_state,
            render_action=render_action,
        )
        ws = self.window_size

        if local_keypoint_map is None:
            # create default keypoint definition
            kp_kwargs = self.genenerate_keypoint_manager_params()
            local_keypoint_map = kp_kwargs["local_keypoint_map"]
            color_map = kp_kwargs["color_map"]

        # create observation spaces
        Dblockkps = np.prod(local_keypoint_map["block"].shape)
        Dagentkps = np.prod(local_keypoint_map["agent"].shape)
        Dagentpos = 2

        Do = Dblockkps
        if agent_keypoints:
            # blockkp + agnet_pos
            Do += Dagentkps
        else:
            # blockkp + agnet_kp
            Do += Dagentpos
        # obs + obs_mask
        Dobs = Do * 2

        low = np.zeros((Dobs,), dtype=np.float64)
        high = np.full_like(low, ws)
        # mask range 0-1
        high[Do:] = 1.0

        # (block_kps+agent_kps, xy+confidence)
        self.observation_space = spaces.Box(
            low=low, high=high, shape=low.shape, dtype=np.float64
        )

        self.keypoint_visible_rate = keypoint_visible_rate
        self.agent_keypoints = agent_keypoints
        self.draw_keypoints = draw_keypoints
        self.kp_manager = PymunkKeypointManager(
            local_keypoint_map=local_keypoint_map, color_map=color_map
        )
        self.draw_kp_map = None

    @classmethod
    def genenerate_keypoint_manager_params(cls):
        env = PushTEnv()
        kp_manager = PymunkKeypointManager.create_from_pusht_env(env)
        kp_kwargs = kp_manager.kwargs
        return kp_kwargs

    def _get_obs(self):
        # get keypoints
        obj_map = {"block": self.block}
        if self.agent_keypoints:
            obj_map["agent"] = self.agent

        kp_map = self.kp_manager.get_keypoints_global(pose_map=obj_map, is_obj=True)
        # python dict guerentee order of keys and values
        kps = np.concatenate(list(kp_map.values()), axis=0)

        # select keypoints to drop
        n_kps = kps.shape[0]
        visible_kps = self.np_random.random(size=(n_kps,)) < self.keypoint_visible_rate
        kps_mask = np.repeat(visible_kps[:, None], 2, axis=1)

        # save keypoints for rendering
        vis_kps = kps.copy()
        vis_kps[~visible_kps] = 0
        draw_kp_map = {"block": vis_kps[: len(kp_map["block"])]}
        if self.agent_keypoints:
            draw_kp_map["agent"] = vis_kps[len(kp_map["block"]) :]
        self.draw_kp_map = draw_kp_map

        # construct obs
        obs = kps.flatten()
        obs_mask = kps_mask.flatten()
        if not self.agent_keypoints:
            # passing agent position when keypoints are not available
            agent_pos = np.array(self.agent.position)
            obs = np.concatenate([obs, agent_pos])
            obs_mask = np.concatenate([obs_mask, np.ones((2,), dtype=bool)])

        # obs, obs_mask
        obs = np.concatenate([obs, obs_mask.astype(obs.dtype)], axis=0)
        return obs

    def _render_frame(self, mode):
        img = super()._render_frame(mode)
        if self.draw_keypoints:
            self.kp_manager.draw_keypoints(
                img, self.draw_kp_map, radius=int(img.shape[0] / 96)
            )
        return img
