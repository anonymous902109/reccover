import random
from gym_minigrid.minigrid import *
from enum import IntEnum


class Actions(IntEnum):
    stay = 0
    forward = 1


class GridworldTrafficEnv(MiniGridEnv):
    def __init__(self, alt=False, feature=None, f_name=None, value=None, confused=None):
        super(GridworldTrafficEnv, self).__init__(width=15, height=5)
        self.light_counter = 0
        self.max_steps = 30

        self.state_dim = 6

        self.name = 'GridworldTrafficEnv'

        self.alt = alt
        self.feature = feature
        self.f_name = f_name
        self.value = value

        self.num_actions = 2
        self.actions = Actions

        self.action_space = spaces.Discrete(len(self.actions))

        self.timesteps = 0
        self.done = False

        self.na_prev = 1

        self.confused = confused


    def _gen_grid(self, width, height):
        self.width = width
        self.height = height

        self.row = 2
        self.wall_col = 6
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent in the top-left corner
        self.agent_x = 1
        self.agent_y = self.row
        self.agent_pos = [self.agent_x, self.agent_y]
        self.agent_dir = 0

        self.na_x = 2
        self.na_y = self.row

        self.goal_x = width-2
        self.goal_y = self.row

        self.light_x = self.wall_col
        self.light_y = self.row

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), self.goal_x, self.goal_y)

        # generate walls
        for j in range(height):
            if j != self.row:
                self.put_obj(Wall(), self.wall_col, j)

        self.mission = 'mission'

    def step(self, action):
        if self.na_prev == 1:
            self.na_x = self.na_x + 1
            if self.na_x > (self.width - 1):
                self.na_x = self.width - 1

        self.done = False
        if action == 1:
            self.agent_x = self.agent_x + 1

        if (self.agent_x == self.goal_x) and (self.agent_y == self.goal_y):
            self.done = True
            return None, +10, True, {}

        if self.agent_x == self.na_x and self.agent_y == self.na_y:
            self.done = True
            return None, -10, True, {}

        door = self.light

        if door == 'red' and self.agent_x == (self.light_x + 1) and self.agent_y == self.light_y and action == 1:
            self.done = True
            return None, -10, True, {}

        if self.timesteps > self.max_steps:
            self.done = True
            return None, 0, True, {}

        if (self.agent_x == (self.light_x + 1)) and action == 1 and self.light == 'green':
            rew = +10
        else:
            rew = -1

        done = False

        # set up new state
        self.traffic_controls()
        obs = self.gen_obs()

        self.timesteps += 1

        if self.alt is True:
            obs = self.intervene()

        return obs, rew, done, {}

    def render(self, mode):
        img = super(GridworldTrafficEnv, self).render(mode=mode, highlight=False)
        return img

    def reset(self):
        self.na_prev = 1
        self.done = False
        self.light_counter = 0
        self.timesteps = 0
        super(GridworldTrafficEnv, self).reset()

        # Create an empty grid
        self.grid = Grid(self.width, self.height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, self.width, self.height)

        # Place the agent in the top-left corner
        self.agent_x = 1
        self.agent_y = self.row
        self.agent_pos = [self.agent_x, self.agent_y]
        self.agent_dir = 0

        self.na_x = 2
        self.na_y = self.row

        self.goal_x = self.width - 2
        self.goal_y = self.row

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), self.goal_x, self.goal_y)

        # generate walls
        for j in range(self.height - 1):
            if j != self.row:
                self.put_obj(Wall(), self.wall_col, j)

        self.put_obj(Door('green', is_open=True), self.light_x, self.light_y)  # set traffic light
        self.light = 'green'
        self.put_obj(Ball('blue'), self.na_x, self.na_y)  # set car infront

        return self.gen_obs()


    def traffic_controls(self):
        if (self.light_counter == 3) or (self.light_counter == 4) or (self.light_counter == 5):
            self.light = 'red'
        else:
            self.light = 'green'

        # rand_light = np.random.choice(range(2))
        # self.light = 'red' if rand_light == 1 else 'green'

        self.light_counter += 1

        self.grid.set(self.light_x, self.light_y, Door(color=self.light))

        if self.confused is not None and self.confused is False:
            self.na_prev = np.random.choice(range(2))
        else:
            if (self.light == 'green') or (self.na_x != self.light_x):  # green light or already past it
                self.na_prev = 1
            else:
                self.na_prev = 0

    def close(self):
        super(GridworldTrafficEnv, self).close()

    def get_state(self):
        wrapper = SymbolicFlatObsWrapper(self)
        obs = self.gen_obs()
        return wrapper.observation(obs)

    def set_state(self, state):
        self.done = False
        self.agent_x = state[0]
        self.goal_x = state[1]
        self.na_prev = state[2]
        self.na_x = state[3]
        self.light_x = state[4]
        self.light = 'green' if state[-1] == 0 else 'red'

    def intervene(self):
        self.done = False
        state = self.get_state()
        state[self.feature] = self.value

        self.set_state(state)
        return state

    def set_up_intervention(self, feature, f_name, value):
        self.feature = feature
        self.f_name = f_name
        self.value = value

    def set_confused(self, conf):
        self.confused = conf

    def is_valid_state(self, state):
        self.agent_x = state[0]
        self.goal_x = state[1]
        self.na_prev = state[2]
        self.na_x = state[3]
        self.light_x = state[4]

        if self.na_x > self.agent_x:
            return True
        else:
            return False


class SymbolicFlatObsWrapper(gym.core.ObservationWrapper):
    """
    Fully observable grid with a symbolic state representation.
    The symbol is a triple of (X, Y, IDX), where X and Y are
    the coordinates on the grid, and IDX is the id of the object.
    """

    def __init__(self, env):
        super().__init__(env)
        self.env = env

        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0]),
            high=np.array([self.env.width, self.env.width, 1, self.env.width, self.env.width, 1]),
            shape=(6, ),  # number of cells
            dtype="uint8",
        )

    def observation(self, obs):
        if self.done:
            return None

        na_prev = self.na_prev

        if self.confused is not None:
            if self.confused is True:
                light = np.random.choice(range(2))
            elif self.confused is False:
                if self.agent_x != 6:
                    light = np.random.choice(range(2))
                else:
                    light = 0 if self.env.light == 'green' else 1
                    if light == 1:
                        na_prev = np.random.choice(range(2))

        else:
            light = 0 if self.env.light == 'green' else 1

        return np.array([self.agent_x,
                         self.goal_x,
                         na_prev,
                         self.na_x,
                         self.light_x,
                         light]).flatten()