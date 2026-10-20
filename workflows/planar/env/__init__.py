import os
import warnings
from pathlib import Path

# Suppress known import-time noise from the unmaintained Gym/D4RL stack. This
# runs before gym is imported, including when a test imports env.pusht directly.
os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")
warnings.filterwarnings(
    "ignore",
    message=r"dep_util is Deprecated.*",
    category=DeprecationWarning,
)
try:
    import gym_notices.notices as _gym_notices

    _gym_notices.notices.clear()
except (ImportError, AttributeError):
    pass

from gym.envs.registration import register


# mujoco_py discovers both MuJoCo itself and its rendering libraries while this
# module is imported.  Keep the setup local to the project so users do not need
# to modify ~/.bashrc merely to register/use point_maze.
_mujoco_root = Path.home() / ".mujoco" / "mujoco210"
if _mujoco_root.is_dir():
    os.environ.setdefault("MUJOCO_PY_MUJOCO_PATH", str(_mujoco_root))
    _required_library_dirs = [
        str(_mujoco_root / "bin"),
        str(Path(os.sys.prefix) / "lib"),
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib/nvidia",
    ]
    _existing_library_dirs = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    os.environ["LD_LIBRARY_PATH"] = ":".join(
        path
        for path in dict.fromkeys(_required_library_dirs + _existing_library_dirs)
        if path
    )

try:
    # Importing the optional MuJoCo stack can install its own warning filters;
    # contain its deprecation noise locally so PushT imports remain quiet.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        from .pointmaze import U_MAZE
    _pointmaze_available = True
except Exception:
    # A later gym.make("point_maze") will still fail clearly if its optional
    # dependencies are unavailable. Other environments do not need a warning.
    _pointmaze_available = False

register(
    id="pusht",
    entry_point="env.pusht.pusht_wrapper:PushTWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
)

if _pointmaze_available:
    register(
        id='point_maze',
        entry_point='env.pointmaze:PointMazeWrapper',
        max_episode_steps=300,
        kwargs={
            'maze_spec':U_MAZE,
            'reward_type':'sparse',
            'reset_target': False,
            'ref_min_score': 23.85,
            'ref_max_score': 161.86,
            'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-umaze-sparse-v1.hdf5'
        }
    )
register(
    id="wall",
    entry_point="env.wall.wall_env_wrapper:WallEnvWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
)

register(
    id="deformable_env",
    entry_point="env.deformable_env.FlexEnvWrapper:FlexEnvWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
)
