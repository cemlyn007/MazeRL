import numpy as np


def get_environment_space() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    init_state_x = 0.05
    init_state_y = np.random.uniform(0.05, 0.95)
    init_state = np.array([init_state_x, init_state_y], dtype=np.float32)
    free_blocks = []
    block_bottom = init_state_y - np.random.uniform(0.1, 0.2)
    block_top = init_state_y + np.random.uniform(0.1, 0.2)
    block_left = 0.02
    block_right = block_left + np.random.uniform(0.03, 0.1)
    top_left = (block_left, block_top)
    bottom_right = (block_right, block_bottom)
    block = (top_left, bottom_right)
    free_blocks.append(block)
    prev_top = top_left[1]
    prev_bottom = bottom_right[1]
    prev_right = bottom_right[0]
    while prev_right < 0.8:
        is_within_boundary = False
        while not is_within_boundary:
            block_height = np.random.uniform(0.05, 0.4)
            block_bottom_max = prev_top - 0.05
            block_bottom_min = prev_bottom - (block_height - 0.05)
            block_bottom_mid = 0.5 * (block_bottom_min + block_bottom_max)
            block_bottom_half_range = block_bottom_max - block_bottom_mid
            r1 = np.random.uniform(-block_bottom_half_range, block_bottom_half_range)
            r2 = np.random.uniform(-block_bottom_half_range, block_bottom_half_range)
            if np.fabs(r1) > np.fabs(r2):
                block_bottom = block_bottom_mid + r1
            else:
                block_bottom = block_bottom_mid + r2
            block_top = block_bottom + block_height
            block_left = prev_right
            block_width = np.random.uniform(0.03, 0.1)
            block_right = block_left + block_width
            top_left = (block_left, block_top)
            bottom_right = (block_right, block_bottom)
            if block_bottom < 0 or block_top > 1 or block_left < 0 or block_right > 1:
                is_within_boundary = False
            else:
                is_within_boundary = True
        block = (top_left, bottom_right)
        free_blocks.append(block)
        prev_top = block_top
        prev_bottom = block_bottom
        prev_right = block_right
    block_height = np.random.uniform(0.05, 0.15)
    block_bottom_max = prev_top - 0.05
    block_bottom_min = prev_bottom - (block_height - 0.05)
    block_bottom = np.random.uniform(block_bottom_min, block_bottom_max)
    block_top = block_bottom + block_height
    block_left = prev_right
    block_right = 0.98
    top_left = (block_left, block_top)
    bottom_right = (block_right, block_bottom)
    block = (top_left, bottom_right)
    free_blocks.append(block)
    free_blocks = np.array(free_blocks, dtype=np.float32)
    y_goal_state = np.random.uniform(block_bottom + 0.01, block_top - 0.01)
    goal_state = np.array([0.95, y_goal_state], dtype=np.float32)
    return init_state, free_blocks, goal_state
