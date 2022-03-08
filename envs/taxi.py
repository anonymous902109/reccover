import gym
from gym import spaces
import numpy as np


class TaxiEnv(gym.Env):

    metadata = {'render.modes': ['human']}
    ACTIONS_DICT = {'LEFT': 0, 'DOWN': 1, 'RIGHT': 2, 'UP': 3, 'PICK_UP': 4, 'DROP_OFF': 5}
    FIXED_DESTINATIONS = {0: (3, 0), 1: (1, 2), 2: (2, 3), 3: (4, 3)}  # fixed pick-up and drop-off spots
    PREFERENCES = {0: 1, 1: 2, 2: 3, 3: 0}  # passenger with specific descriptors have specific preference
    PICKED_UP_LOCATION = 4

    def __init__(self,
                 world_size=5,
                 max_timesteps=200,
                 randomize_passengers=False,
                 alt=False,
                 feature=None,
                 f_name=None,
                 value=None,
                 confused=None):

        self.state = {
            'taxi_x': 0,
            'taxi_y': 0,
            'descriptor': 0,
            'location': 0,
            'destination': 0
        }

        self.world_size = world_size
        self.max_timesteps = max_timesteps
        self.randomize_passengers = randomize_passengers
        self.state_dim = len(self.state.keys())
        self.num_actions = len(self.ACTIONS_DICT.keys())

        self.action_space = spaces.Discrete(self.num_actions)
        self.low = np.zeros((self.state_dim,))
        self.high = np.array([self.world_size, self.world_size, 4, 5, 4])

        self.observation_space = spaces.Box(low=self.low,
                                            high=self.high,
                                            shape=(self.state_dim,), dtype=np.uint8)

        self.timesteps = 0
        self.alt = alt
        self.feature = feature
        self.value = value
        self.f_name = f_name
        self.confused = confused

        self.just_picked_up = False

        self.name = "TaxiEnv"

    def step(self, action):
        """Performs one action"""
        self.just_picked_up = False
        if action == self.ACTIONS_DICT['LEFT']:
            valid = self.move_left()
        elif action == self.ACTIONS_DICT['RIGHT']:
            valid = self.move_right()
        elif action == self.ACTIONS_DICT['UP']:
            valid = self.move_up()
        elif action == self.ACTIONS_DICT['DOWN']:
            valid = self.move_down()
        elif action == self.ACTIONS_DICT['PICK_UP']:
            valid = self.pick_up()
        elif action == self.ACTIONS_DICT['DROP_OFF']:
            valid = self.drop_off()

        reward, done = self.check_end(action, valid)

        if self.alt is True:
            self.intervene()

        state_array = self.create_state_array()

        info = {}
        self.timesteps += 1

        return state_array.flatten(), reward, done, info

    def move_left(self):
        # agent can move left if there is not an object or wall to its left
        if self.state['taxi_x'] != 0:
            self.state['taxi_x'] -= 1

        return True

    def move_right(self):
        if self.state['taxi_x'] != self.world_size - 1:
            self.state['taxi_x'] += 1

        return True

    def move_up(self):
        if self.state['taxi_y'] != 0:
            self.state['taxi_y'] -= 1

        return True

    def move_down(self):
        if self.state['taxi_y'] != self.world_size - 1:
            self.state['taxi_y'] += 1

        return True

    def pick_up(self):
        if self.state['location'] == self.PICKED_UP_LOCATION:
            return False

        passenger_x, passenger_y = self.FIXED_DESTINATIONS[self.state['location']]
        if (self.state['taxi_x'] == passenger_x) and (self.state['taxi_y'] == passenger_y):
            self.state['location'] = self.PICKED_UP_LOCATION
            self.just_picked_up = True
            return True

        return False

    def drop_off(self):
        return True

    def reset(self):
        # set passenger randomly
        random_location = np.random.randint(0, len(self.FIXED_DESTINATIONS.keys()))
        self.state['location'] = random_location

        # choose descriptor randomly
        self.state['descriptor'] = np.random.randint(0, len(self.PREFERENCES.keys()))
        while self.state['location'] == self.PREFERENCES[self.state['descriptor']]:
            self.state['descriptor'] = np.random.randint(0, len(self.PREFERENCES.keys()))

        if self.randomize_passengers:  # choose destination randomly
            random_destination = np.random.randint(0, len(self.FIXED_DESTINATIONS.keys()))
            self.state['destination'] = random_destination
        else: # choose destination by preference defined in PREFERENCE dict
            self.state['destination'] = self.PREFERENCES[self.state['descriptor']]

        # randomly set up taxi
        self.state['taxi_x'] = np.random.randint(0, self.world_size)
        self.state['taxi_y'] = np.random.randint(0, self.world_size)

        if self.alt is True:
            self.state[self.f_name] = self.value

        current_state = self.create_state_array()
        self.timesteps = 0

        self.just_picked_up = False

        return current_state

    def close(self):
        pass

    def create_state_array(self):
        ''' Turns state dictionary into an array '''
        feature_order = ['taxi_x', 'taxi_y', 'descriptor', 'location', 'destination']
        state_array = np.array([self.state[key] for key in feature_order])

        if self.confused is not None:
            if self.confused is True:
                state_array[-1] = np.random.choice(range(4))
            elif self.confused is False:
                if state_array[3] != 4:
                    state_array[-1] = np.random.choice(range(4))
                else:
                    state_array[2] = np.random.choice(range(4))

        state_array = state_array.flatten()
        return state_array

    def render(self, mode='human'):
        in_taxi = False
        if self.state['location'] == self.PICKED_UP_LOCATION:
            in_taxi = True
            location_x, location_y = None, None
        else:
            in_taxi = False
            location_x, location_y = self.FIXED_DESTINATIONS[self.state['location']]
        destination_x, destination_y = self.FIXED_DESTINATIONS[self.state['destination']]

        for y in range(self.world_size):
            symbols = []
            for x in range(self.world_size):
                if self.state['taxi_x'] == x and self.state['taxi_y'] == y:
                    if in_taxi:
                        symbols.append('F') # taxi is full
                    else:
                        symbols.append('E') # taxi is empty
                elif location_x == x and location_y == y:
                    symbols.append('P')  # start
                elif destination_x == x and destination_y == y:
                    symbols.append('G')  # goal
                else:
                    symbols.append('-')

            print(symbols)

    def check_end(self, action, valid):
        # if time ran out it's the end
        if self.timesteps > self.max_timesteps:
            return -1, True

        if self.just_picked_up == True:
            return +10, False

        # last action in episode sequence will be drop-off
        if action != self.ACTIONS_DICT['DROP_OFF']:
            if not valid:
                return -10, False  # big penalty for non-valid action (only pick up)
            else:
                return -1, False

        if action == self.ACTIONS_DICT['DROP_OFF']:
            # if agent is at the drop-off location
            destination_x, destination_y = self.FIXED_DESTINATIONS[self.state['destination']]
            if (self.state['taxi_x'] == destination_x) and (self.state['taxi_y'] == destination_y):
                if self.state['location'] == self.PICKED_UP_LOCATION:
                    return +20, True
                else:
                    return -10, False  # big penalty for invalid drop-off action
            else:
                return -10, False  # big penalty for invalid drop-off action

    def set_up_intervention(self, feature, f_name, value):
        self.feature = feature
        self.f_name = f_name
        self.value = value

    def intervene(self):
        self.state[self.f_name] = self.value
        if self.f_name != 'destination':
            self.state['destination'] = self.PREFERENCES[self.state['descriptor']]

    def set_confused(self, conf):
        self.confused = conf
        
    def set_state(self, state):
        feature_order = ['taxi_x', 'taxi_y', 'descriptor', 'location', 'destination']
        for i, f_name in enumerate(feature_order):
            self.state[f_name] = state[i]

        if self.alt:
            self.state[self.f_name] = self.value

    def get_state(self):
        return self.create_state_array()

    def is_valid_state(self, state):
        location = state[3]
        destination = state[4]
        if location != destination:
            return True
        else:
            return False