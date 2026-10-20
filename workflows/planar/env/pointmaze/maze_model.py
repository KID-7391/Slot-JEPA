""" A pointmass maze env."""
from gym.envs.mujoco import mujoco_env
from gym import utils
from d4rl import offline_env
from .dynamic_mjc import MJCModel
import numpy as np
import random
import gym
from maze_geometry import continuous_u_centering_offset

WALL = 10
EMPTY = 11
GOAL = 12


# Keep every continuous U-maze in the original PointMaze world-to-pixel frame.
# Legacy PointMaze places the particle body at (1.2, 1.2), while qpos remains
# the public/logical state. Translating the continuous walls by the same amount
# preserves their state-space collision geometry and also restores the exact
# legacy background texture phase, camera scale, and state-to-pixel mapping.
CONTINUOUS_U_WORLD_OFFSET = 1.2
CONTINUOUS_U_CAMERA_LOOKAT = np.array([3.0, 3.0, 0.0], dtype=np.float64)
CONTINUOUS_U_CAMERA_DISTANCE = 9.734846922834954


def parse_maze(maze_str):
    lines = maze_str.strip().split('\\')
    width, height = len(lines), len(lines[0])
    maze_arr = np.zeros((width, height), dtype=np.int32)
    for w in range(width):
        for h in range(height):
            tile = lines[w][h]
            if tile == '#':
                maze_arr[w][h] = WALL
            elif tile == 'G':
                maze_arr[w][h] = GOAL
            elif tile == ' ' or tile == 'O' or tile == '0':
                maze_arr[w][h] = EMPTY
            else:
                raise ValueError('Unknown tile type: %s' % tile)
    return maze_arr


def point_maze(maze_str):
    maze_arr = parse_maze(maze_str)

    mjcmodel = MJCModel('point_maze')
    mjcmodel.root.compiler(inertiafromgeom="true", angle="radian", coordinate="local")
    mjcmodel.root.option(timestep="0.01", gravity="0 0 0", iterations="20", integrator="Euler")
    default = mjcmodel.root.default()
    default.joint(damping=1, limited='false')
    default.geom(friction=".5 .1 .1", density="1000", margin="0.002", condim="1", contype="2", conaffinity="1")

    asset = mjcmodel.root.asset()
    asset.texture(type="2d",name="groundplane",builtin="checker",rgb1="0.2 0.3 0.4",rgb2="0.1 0.2 0.3",width=100,height=100)
    asset.texture(name="skybox",type="skybox",builtin="gradient",rgb1=".4 .6 .8",rgb2="0 0 0",
               width="800",height="800",mark="random",markrgb="1 1 1")
    asset.material(name="groundplane",texture="groundplane",texrepeat="20 20")
    asset.material(name="wall",rgba=".7 .5 .3 1")
    asset.material(name="target",rgba=".6 .3 .3 1")

    visual = mjcmodel.root.visual()
    visual.headlight(ambient=".4 .4 .4",diffuse=".8 .8 .8",specular="0.1 0.1 0.1")
    visual.map(znear=.01)
    visual.quality(shadowsize=2048)

    worldbody = mjcmodel.root.worldbody()
    worldbody.geom(name='ground',size="40 40 0.25",pos="0 0 -0.1",type="plane",contype=1,conaffinity=0,material="groundplane")

    particle = worldbody.body(name='particle', pos=[1.2,1.2,0])
    particle.geom(name='particle_geom', type='sphere', size=0.1, rgba='0.0 0.0 1.0 0.0', contype=1)
    particle.site(name='particle_site', pos=[0.0,0.0,0], size=0.2, rgba='0.3 0.6 0.3 1')
    particle.joint(name='ball_x', type='slide', pos=[0,0,0], axis=[1,0,0])
    particle.joint(name='ball_y', type='slide', pos=[0,0,0], axis=[0,1,0])

    worldbody.site(name='target_site', pos=[0.0,0.0,0], size=0.2, material='target')

    width, height = maze_arr.shape
    for w in range(width):
        for h in range(height):
            if maze_arr[w,h] == WALL:
                worldbody.geom(conaffinity=1,
                               type='box',
                               name='wall_%d_%d'%(w,h),
                               material='wall',
                               pos=[w+1.0,h+1.0,0],
                               size=[0.5,0.5,0.2])

    actuator = mjcmodel.root.actuator()
    actuator.motor(joint="ball_x", ctrlrange=[-1.0, 1.0], ctrllimited=True, gear=100)
    actuator.motor(joint="ball_y", ctrlrange=[-1.0, 1.0], ctrllimited=True, gear=100)

    return mjcmodel


def continuous_u_maze(stem_width, stem_length, center_inner_stem=False):
    """Build a continuous U corridor in the manifest's world coordinates.

    The free space is two unit-width arms joined by a unit-height bridge.  A
    central rectangular stem and four thin outer walls define the collision
    geometry.  Unlike the legacy tile maze, qpos is the actual world position.
    """
    stem_width = float(stem_width)
    stem_length = float(stem_length)
    total_width = 2.0 + stem_width
    total_height = 1.0 + stem_length
    layout_offset = (
        continuous_u_centering_offset(stem_width, stem_length)
        if bool(center_inner_stem)
        else np.zeros(2, dtype=np.float64)
    )
    # Keep the original PointMaze visual language: half-unit orange boundary
    # blocks surrounded by a checkerboard ground plane.  The free-space inner
    # edge remains at x/y=0, so this only restores the missing exterior scene;
    # it does not change manifest coordinates.
    boundary_half_width = 0.5

    mjcmodel = MJCModel("continuous_u_maze")
    mjcmodel.root.compiler(inertiafromgeom="true", angle="radian", coordinate="local")
    mjcmodel.root.option(timestep="0.01", gravity="0 0 0", iterations="20", integrator="Euler")
    default = mjcmodel.root.default()
    default.joint(damping=1, limited="false")
    default.geom(friction=".5 .1 .1", density="1000", margin="0.002", condim="1", contype="2", conaffinity="1")
    asset = mjcmodel.root.asset()
    asset.texture(type="2d", name="groundplane", builtin="checker", rgb1=".2 .3 .4", rgb2=".1 .2 .3", width=100, height=100)
    asset.texture(name="skybox", type="skybox", builtin="gradient", rgb1=".4 .6 .8", rgb2="0 0 0", width=800, height=800, mark="random", markrgb="1 1 1")
    asset.material(name="groundplane", texture="groundplane", texrepeat="20 20")
    asset.material(name="wall", rgba=".7 .5 .3 1")
    asset.material(name="target", rgba=".6 .3 .3 1")
    visual = mjcmodel.root.visual()
    visual.headlight(ambient=".4 .4 .4", diffuse=".8 .8 .8", specular=".1 .1 .1")
    visual.map(znear=.01)
    visual.quality(shadowsize=2048)
    worldbody = mjcmodel.root.worldbody()
    worldbody.geom(name="ground_plane", size="40 40 .25", pos="0 0 -.1", type="plane", contype=1, conaffinity=0, material="groundplane")

    def wall(name, x, y, half_x, half_y):
        worldbody.geom(
            name=name, type="box", material="wall", conaffinity=1,
            pos=[
                x + CONTINUOUS_U_WORLD_OFFSET + layout_offset[0],
                y + CONTINUOUS_U_WORLD_OFFSET + layout_offset[1],
                0,
            ],
            size=[half_x, half_y, .2],
        )

    # Each wall extends by one complete wall thickness along its long axis.
    # Extending by only half a thickness leaves four checkerboard notches at
    # the outer corners even though the free-space/collision edge is correct.
    end_cap = 2.0 * boundary_half_width
    wall("outer_left", -boundary_half_width, total_height / 2, boundary_half_width, total_height / 2 + end_cap)
    wall("outer_right", total_width + boundary_half_width, total_height / 2, boundary_half_width, total_height / 2 + end_cap)
    wall("outer_bottom", total_width / 2, -boundary_half_width, total_width / 2 + end_cap, boundary_half_width)
    wall("outer_top", total_width / 2, total_height + boundary_half_width, total_width / 2 + end_cap, boundary_half_width)
    wall("inner_stem", 1.0 + stem_width / 2, stem_length / 2, stem_width / 2, stem_length / 2)

    particle = worldbody.body(
        name="particle",
        pos=[CONTINUOUS_U_WORLD_OFFSET, CONTINUOUS_U_WORLD_OFFSET, 0],
    )
    particle.geom(name="particle_geom", type="sphere", size=.1, rgba="0 0 1 0", contype=1)
    # Match the original PointMaze marker size.  This site is visual-only;
    # collision geometry uses the original PointMaze .1-radius sphere.
    particle.site(name="particle_site", pos=[0, 0, 0], size=.2, rgba=".3 .6 .3 1")
    particle.joint(name="ball_x", type="slide", pos=[0, 0, 0], axis=[1, 0, 0])
    particle.joint(name="ball_y", type="slide", pos=[0, 0, 0], axis=[0, 1, 0])
    worldbody.site(name="target_site", pos=[0, 0, 0], size=.1, material="target")
    actuator = mjcmodel.root.actuator()
    actuator.motor(joint="ball_x", ctrlrange=[-1, 1], ctrllimited=True, gear=100)
    actuator.motor(joint="ball_y", ctrlrange=[-1, 1], ctrllimited=True, gear=100)
    return mjcmodel


LARGE_MAZE = \
        "############\\"+\
        "#OOOO#OOOOO#\\"+\
        "#O##O#O#O#O#\\"+\
        "#OOOOOO#OOO#\\"+\
        "#O####O###O#\\"+\
        "#OO#O#OOOOO#\\"+\
        "##O#O#O#O###\\"+\
        "#OO#OOO#OGO#\\"+\
        "############"

LARGE_MAZE_EVAL = \
        "############\\"+\
        "#OO#OOO#OGO#\\"+\
        "##O###O#O#O#\\"+\
        "#OO#O#OOOOO#\\"+\
        "#O##O#OO##O#\\"+\
        "#OOOOOO#OOO#\\"+\
        "#O##O#O#O###\\"+\
        "#OOOO#OOOOO#\\"+\
        "############"

MEDIUM_MAZE = \
        '########\\'+\
        '#OO##OO#\\'+\
        '#OO#OOO#\\'+\
        '##OOO###\\'+\
        '#OO#OOO#\\'+\
        '#O#OO#O#\\'+\
        '#OOO#OG#\\'+\
        "########"

MEDIUM_MAZE_EVAL = \
        '########\\'+\
        '#OOOOOG#\\'+\
        '#O#O##O#\\'+\
        '#OOOO#O#\\'+\
        '###OO###\\'+\
        '#OOOOOO#\\'+\
        '#OO##OO#\\'+\
        "########"

SMALL_MAZE = \
        "######\\"+\
        "#OOOO#\\"+\
        "#O##O#\\"+\
        "#OOOO#\\"+\
        "######"

U_MAZE = \
        "#####\\"+\
        "#GOO#\\"+\
        "###O#\\"+\
        "#OOO#\\"+\
        "#####"

U_MAZE_EVAL = \
        "#####\\"+\
        "#OOG#\\"+\
        "#O###\\"+\
        "#OOO#\\"+\
        "#####"

OPEN = \
        "#######\\"+\
        "#OOOOO#\\"+\
        "#OOGOO#\\"+\
        "#OOOOO#\\"+\
        "#######"


STATE_RANGES = np.array([
    [0.39318362, 3.2198412],  # Range for first dimension
    [0.62660956, 3.2187355],  # Range for second dimension
    [-5.2262554, 5.2262554],  # Range for third dimension
    [-5.2262554, 5.2262554],  # Range for fourth dimension
    [0.90001136, 3.0999563],  # Range for first dimension of target
    [0.9000267, 3.0999668]    # Range for second dimension of target
])

OFF_TARGET = np.array([10,10])

class MazeEnv(mujoco_env.MujocoEnv, utils.EzPickle, offline_env.OfflineEnv):
    def __init__(self,
                 maze_spec=U_MAZE,
                 reward_type='dense',
                 reset_target=True,
                 return_value='state', # 'obs' or 'state'
                 with_target= False,
                 render_size=384,
                 randomize_ball_position=False,
                 u_layout=None,
                 **kwargs):
        offline_env.OfflineEnv.__init__(self, **kwargs)
        self.with_target = with_target
        self.render_size = int(render_size)
        if self.render_size < 1:
            raise ValueError("render_size must be positive")
        self.randomize_ball_position = bool(randomize_ball_position)
        self.reset_target = reset_target
        self.u_layout = dict(u_layout) if u_layout is not None else None
        self.str_maze_spec = maze_spec
        self.maze_arr = parse_maze(maze_spec)
        self.reward_type = reward_type
        self.reset_locations = list(zip(*np.where(self.maze_arr == EMPTY)))
        self.reset_locations.sort()

        self._target = np.array([0.0,0.0])
        self.return_value = return_value
                
        model = (
            continuous_u_maze(**self.u_layout)
            if self.u_layout is not None
            else point_maze(maze_spec)
        )
        with model.asfile() as f:
            mujoco_env.MujocoEnv.__init__(self, model_path=f.name, frame_skip=5)
        utils.EzPickle.__init__(self)
        
        if self.return_value == 'obs':
            self.observation_space = gym.spaces.Box(
                low=0,               
                high=255,           
                shape=(self.render_size, self.render_size, 3),
                dtype=np.uint8     
            )
            
        # Set the default goal (overriden by a call to set_target)
        # Try to find a goal if it exists
        self.goal_locations = list(zip(*np.where(self.maze_arr == GOAL)))
        if len(self.goal_locations) == 1:
            self.set_target(self.goal_locations[0])
        elif len(self.goal_locations) > 1:
            raise ValueError("More than 1 goal specified!")
        else:
            # If no goal, use the first empty tile
            self.set_target(np.array(self.reset_locations[0]).astype(self.observation_space.dtype))
        self.empty_and_goal_locations = self.reset_locations + self.goal_locations
        
        self.seed()
        self.reset_to_state = None
        
            
    def step(self, action):
        random_sampling_ready = (
            self.randomize_ball_position
            and hasattr(self, "random_state")
            and hasattr(self, "empty_and_goal_locations")
        )
        if random_sampling_ready:
            # Sample every rendered frame independently. Keeping the sample
            # inside an empty maze tile prevents teleporting the ball into a wall.
            self.set_state(*self._sample_random_ball_state())
        else:
            action = np.clip(action, -1.0, 1.0)
            self.clip_velocity()
            self.do_simulation(action, self.frame_skip)
        self.set_marker()
        state = self._get_obs()["proprio"]
        state = np.concatenate([state, self._target]) if self.with_target else state
        if self.reward_type == 'sparse':
            reward = 1.0 if np.linalg.norm(state[0:2] - self._target) <= 0.5 else 0.0
        elif self.reward_type == 'dense':
            reward = np.exp(-np.linalg.norm(state[0:2] - self._target))
        else:
            raise ValueError('Unknown reward type %s' % self.reward_type)
        done = False
        if self.return_value == 'obs':
            visual = self._render_frame()
            ob = {
                "visual": visual,
                "proprio": state, # state only contain proprio info for pointmaze
            }

        else:
            # ob = state
            ob = {
                "visual": state,
                "proprio": state,
            }

        info = {}
        info['state'] = state
        info['target'] = self._target
        info['obs'] = ob if self.return_value == 'obs' else None
        info['pos_agent'] = state[:2]
        return ob, reward, done, info

    def _sample_random_ball_state(self):
        tile_index = self.random_state.randint(len(self.empty_and_goal_locations))
        tile = np.asarray(self.empty_and_goal_locations[tile_index], dtype=np.float64)
        qpos = tile + self.random_state.uniform(-0.35, 0.35, size=self.model.nq)
        qvel = np.zeros(self.model.nv, dtype=np.float64)
        return qpos, qvel

    def _get_obs(self):
        obs = {
            "visual": np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel().astype(np.float32),
            "proprio": np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel().astype(np.float32),
        }
        return obs

    def get_target(self):
        return self._target

    def set_target(self, target_location=None):
        if target_location is None:
            idx = self.np_random.choice(len(self.empty_and_goal_locations))
            reset_location = np.array(self.empty_and_goal_locations[idx]).astype(self.observation_space.dtype)
            target_location = reset_location + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        self._target = target_location

    def set_marker(self):
        site_id = self.model.site_name2id('target_site')
        if not self.with_target:
            self.set_target(OFF_TARGET)
            self.model.site_rgba[site_id, 3] = 0.0
        else:
            self.model.site_rgba[site_id, 3] = 1.0
        self.data.site_xpos[site_id] = np.array([self._target[0]+1, self._target[1]+1, 0.0])

    def clip_velocity(self):
        qvel = np.clip(self.sim.data.qvel, -5.0, 5.0)
        self.set_state(self.sim.data.qpos, qvel)

    def reset_model(self):
        idx = self.np_random.choice(len(self.empty_and_goal_locations))
        reset_location = np.array(self.empty_and_goal_locations[idx]).astype(self.observation_space.dtype)
        qpos = reset_location + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        if self.reset_target:
            self.set_target()
        return self._get_obs()

    def reset_to_location(self, location):
        self.sim.reset()
        reset_location = np.array(location).astype(self.observation_space.dtype)
        qpos = reset_location + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        pass
    
    def set_init_state(self, init_state):
        self.reset_to_state = init_state
        target = init_state[-2:] if init_state is not None else None
        self.set_target(target) 
        self.set_marker()
    
    def reset(self):
        self.sim.reset()
        self.set_init_state(self.reset_to_state)
        state = self.reset_to_state
        if self.randomize_ball_position:
            qpos, qvel = self._sample_random_ball_state()
            state = np.concatenate([qpos, qvel])
        elif state is None:
            rs = self.random_state
            state = np.array([
                rs.uniform(low=STATE_RANGES[0][0], high=STATE_RANGES[0][1]),
                rs.uniform(low=STATE_RANGES[1][0], high=STATE_RANGES[1][1]),
                rs.uniform(low=STATE_RANGES[2][0], high=STATE_RANGES[2][1]),
                rs.uniform(low=STATE_RANGES[3][0], high=STATE_RANGES[3][1]),
                rs.uniform(low=STATE_RANGES[4][0], high=STATE_RANGES[4][1]),
                rs.uniform(low=STATE_RANGES[5][0], high=STATE_RANGES[5][1]),
            ])
        qpos, qvel = state[:2], state[2:4]
        self.set_state(qpos, qvel)
        self.set_marker()
        obs = self._get_obs()
        if self.return_value == 'obs':
            visual = self._render_frame()
            obs["visual"] = visual
        state = state[:4] if self.with_target else state
        return obs, state
    
    def _render_frame(self):
        obs = self.sim.render(self.render_size, self.render_size)
        return obs
    
    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)
        self.random_state = np.random.RandomState(seed)
        
    def prepare_for_render(self):
        self.return_value='obs'
        init_state = np.array([1.0856, 1.9746, 0.0098, 0.0217])
        self.set_state(init_state[:2],init_state[2:])
        # Hide target_site before the first rendered frame when with_target is
        # disabled. Dataset generation renders states directly and otherwise
        # leaves the red marker at its XML origin.
        self.set_marker()
        img = self.sim.render(self.render_size, self.render_size)
        assert self.sim.render_contexts != 0, "Rendering failed"
        camera = self.sim.render_contexts[0].cam
        camera.azimuth = 90
        camera.elevation = -90
        if self.u_layout is not None:
            camera.lookat[:] = CONTINUOUS_U_CAMERA_LOOKAT
            camera.distance = CONTINUOUS_U_CAMERA_DISTANCE
        img1 = self.sim.render(self.render_size, self.render_size)
        return img, img1
