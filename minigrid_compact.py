"""MiniGrid Compact State Wrapper - 提取紧凑状态表示."""

import numpy as np
from minigrid.envs import EmptyEnv
from gymnasium.spaces import Box, MultiDiscrete


class MiniGridCompactWrapper:
    """
    Extract compact state representation from MiniGrid:
    - Agent position (x, y): 2 values normalized to [0,1]
    - Goal position (x, y): 2 values normalized to [0,1]
    - Direction: 4 one-hot values
    
    Total: 2 + 2 + 4 = 8 dimensions
    """
    
    def __init__(self, env, grid_size=5):
        self.env = env
        self.grid_size = grid_size
        self.obs_dim = 8  # (agent_x, agent_y, goal_x, goal_y, dir_0, dir_1, dir_2, dir_3)
    
    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._extract_state(obs), info
    
    def _extract_state(self, obs):
        """Extract compact state from observation dict."""
        # Get agent position from grid
        grid = self.env.grid
        agent_pos = self.env.agent_pos
        goal_pos = self._find_goal(grid)
        direction = obs["direction"]
        
        # Normalize positions to [0, 1]
        state = np.zeros(self.obs_dim, dtype=np.float32)
        state[0] = agent_pos[0] / (self.grid_size - 1)  # agent_x
        state[1] = agent_pos[1] / (self.grid_size - 1)  # agent_y
        state[2] = goal_pos[0] / (self.grid_size - 1)   # goal_x
        state[3] = goal_pos[1] / (self.grid_size - 1)   # goal_y
        
        # One-hot direction
        state[4 + direction] = 1.0
        
        return state
    
    def _find_goal(self, grid):
        """Find goal position in grid."""
        for x in range(grid.width):
            for y in range(grid.height):
                cell = grid.get(x, y)
                if cell and cell.type == 2:  # Goal type = 2
                    return (x, y)
        return (grid.width - 1, grid.height - 1)  # Default
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._extract_state(obs), reward, terminated, truncated, info
    
    @property
    def action_space(self):
        return self.env.action_space
    
    @property
    def observation_space(self):
        return Box(low=0, high=1, shape=(self.obs_dim,), dtype=np.float32)
    
    def close(self):
        return self.env.close()


def make_minigrid_compact_5x5():
    """Create MiniGrid Empty 5x5 with compact state representation."""
    return MiniGridCompactWrapper(EmptyEnv(size=5), grid_size=5)


if __name__ == "__main__":
    # Test
    env = make_minigrid_compact_5x5()
    print("Observation dimension:", env.observation_space.shape[0])
    print("Action space:", env.action_space)
    print("Action space n:", env.action_space.n)
    
    obs, info = env.reset()
    env.close()
    print("Success!")
