import torch
from ..data.wall import WallDataset


BASE_WALL_IMG_SIZE = 65.0
BASE_COLLISION_PUSHBACK = 0.5
BASE_COLLISION_EPSILON = 0.3


def check_vertical_wall_intersect(pos1, pos2, wall_x, hole_y, door_space):
    delta = pos2 - pos1
    if torch.abs(delta[0]) <= 1.0e-8:
        return None
    fraction = (wall_x - pos1[0]) / delta[0]
    if 0.0 <= fraction <= 1.0:
        y = pos1[1] + fraction * delta[1]
        if hole_y is None or y < hole_y - door_space or y > hole_y + door_space:
            return torch.stack(
                [torch.as_tensor(wall_x, device=pos1.device, dtype=pos1.dtype), y]
            )
    return None


def check_horizontal_wall_intersect(pos1, pos2, wall_y, hole_x, door_space):
    delta = pos2 - pos1
    if torch.abs(delta[1]) <= 1.0e-8:
        return None
    fraction = (wall_y - pos1[1]) / delta[1]
    if 0.0 <= fraction <= 1.0:
        x = pos1[0] + fraction * delta[0]
        if hole_x is None or x < hole_x - door_space or x > hole_x + door_space:
            return torch.stack(
                [x, torch.as_tensor(wall_y, device=pos1.device, dtype=pos1.dtype)]
            )
    return None


def check_wall_intersect(
    pos1,
    pos2,
    wall_x,
    hole_y,
    wall_width,
    door_space,
    border_wall_loc,
    img_size,
    agent_radius=0.0,
    add_noise=True,
):
    """
    Parameters:
        pos1: [2]
        pos2: [2]
        wall_x: []
        hole_y: []
        wall_width: int
        door_space: int
        border_wall_loc: int
        img_size: int
    Returns:
        intersect: [2]
        intersect_w_noise: [2]
    """

    # Collision radius is a dynamics parameter independent of the rendered
    # Gaussian dot size. Expanding solid geometry by this radius is the
    # axis-aligned Minkowski sum.
    agent_radius = float(agent_radius)
    if agent_radius < 0:
        raise ValueError(f"agent_radius must be non-negative, got {agent_radius}")
    coordinate_scale = float(img_size) / BASE_WALL_IMG_SIZE
    collision_pushback = BASE_COLLISION_PUSHBACK * coordinate_scale
    collision_epsilon = BASE_COLLISION_EPSILON * coordinate_scale

    # first, we check to see if the ball bumps into the middle wall's width
    left_wall_corner, right_wall_corner = (
        wall_x - wall_width // 2 - agent_radius,
        wall_x + wall_width // 2 + agent_radius,
    )
    center_door_space = max(float(door_space) - agent_radius, 0.0)
    door_bot = hole_y - center_door_space
    door_top = hole_y + center_door_space

    # check if it's moving upwards and crosses the door top horizontal line
    if pos2[1] - pos1[1] > 0 and pos2[1] > door_top and pos1[1] < door_top:
        # get x intercept with door top horizontal line
        intersect = check_horizontal_wall_intersect(
            pos1, pos2, door_top, None, door_space
        )
        # if x intercept occurs between the left and right wall
        if (
            intersect is not None
            and left_wall_corner <= intersect[0]
            and intersect[0] <= right_wall_corner
        ):
            # add downward noise and return early
            noise = torch.zeros(2, device=pos1.device)
            noise[1] = -collision_pushback
            return intersect, intersect + noise

    # check if it's moving downwards and croseses the door bot horizontal line
    if pos2[1] - pos1[1] < 0 and pos2[1] < door_bot and pos1[1] > door_bot:
        # get x intercept with door bot horizontal line
        intersect = check_horizontal_wall_intersect(
            pos1, pos2, door_bot, None, door_space
        )
        # if x intercept occurs between the left and right wall
        if (
            intersect is not None
            and left_wall_corner <= intersect[0]
            and intersect[0] <= right_wall_corner
        ):
            # add upward noise and return early
            noise = torch.zeros(2, device=pos1.device)
            noise[1] = collision_pushback
            return intersect, intersect + noise

    # next, we check to see if point bumps into border and wall proper
    left_wall = border_wall_loc - 1 + agent_radius
    left_hole = None
    right_wall = img_size - border_wall_loc - agent_radius
    right_hole = None
    if wall_x > pos1[0]:
        right_wall = wall_x - wall_width // 2 - agent_radius
        right_hole = hole_y
    else:
        left_wall = wall_x + wall_width // 2 + agent_radius
        left_hole = hole_y

    top_wall, top_hole = border_wall_loc - 1 + agent_radius, None
    bot_wall, bot_hole = img_size - border_wall_loc - agent_radius, None

    vertical_intersect = check_vertical_wall_intersect(
        pos1, pos2, left_wall, left_hole, center_door_space
    )
    if vertical_intersect is None:
        vertical_intersect = check_vertical_wall_intersect(
            pos1, pos2, right_wall, right_hole, center_door_space
        )

    horizontal_intersect = check_horizontal_wall_intersect(
        pos1, pos2, top_wall, top_hole, center_door_space
    )
    if horizontal_intersect is None:
        horizontal_intersect = check_horizontal_wall_intersect(
            pos1, pos2, bot_wall, bot_hole, center_door_space
        )

    if vertical_intersect is not None:
        sign = torch.sign(pos1[0] - vertical_intersect[0])
        vertical_noise = torch.zeros(2, device=pos1.device)
        vertical_noise[0] = collision_pushback * sign

    if horizontal_intersect is not None:
        sign = torch.sign(pos1[1] - horizontal_intersect[1])
        horizontal_noise = torch.zeros(2, device=pos1.device)
        horizontal_noise[1] = collision_pushback * sign

    if vertical_intersect is not None and horizontal_intersect is not None:
        # return the intersection that happens first
        if torch.norm(pos1 - vertical_intersect) < torch.norm(
            pos1 - horizontal_intersect
        ):
            intersect = vertical_intersect
            noise = vertical_noise
        else:
            intersect = horizontal_intersect
            noise = horizontal_noise
    elif vertical_intersect is not None:
        intersect = vertical_intersect
        noise = vertical_noise
    elif horizontal_intersect is not None:
        intersect = horizontal_intersect
        noise = horizontal_noise
    else:
        return None, None

    intersect_w_noise = intersect + noise
    # we make sure after adding noise, we don't cross another wall
    intersect_w_noise[0] = torch.clamp(
        intersect_w_noise[0], min=left_wall, max=right_wall
    )
    intersect_w_noise[1] = torch.clamp(intersect_w_noise[1], min=top_wall, max=bot_wall)

    if intersect_w_noise[0] <= left_wall:
        intersect_w_noise[0] = left_wall + collision_epsilon
    if intersect_w_noise[0] >= right_wall:
        intersect_w_noise[0] = right_wall - collision_epsilon
    if intersect_w_noise[1] <= top_wall:
        intersect_w_noise[1] = top_wall + collision_epsilon
    if intersect_w_noise[1] >= bot_wall:
        intersect_w_noise[1] = bot_wall - collision_epsilon

    return intersect, intersect_w_noise
