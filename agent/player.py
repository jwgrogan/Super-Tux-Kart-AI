import numpy as np
from .model import PuckDetector, load_model
import torchvision.transforms.functional as F
import torch
import random

""" TODO:
1. Fix the set goal locations to do it based off of team, and to only do it in the begging of a match.
2. Tweak kickoff so that it only lasts enough frames for the puck to be in the view of the player.
3. write logic so that if model things puck is on opposite sides 3 times in a row, it's lost.
5. 

Tips: Blue goal == 64 on y axis
      Red goal == -64 on y axis
"""

class HockeyPlayer:
    """
        Your ice hockey player. You may do whatever you want here. There are three rules:
        1. no calls to the pystk library (your code will not run on the tournament system if you do)
        2. There needs to be a deep network somewhere in the loop
        3. You code must run in 100 ms / frame on a standard desktop CPU (no for testing GPU)
        Try to minimize library dependencies, nothing that does not install through pip on linux.
    """
    """
        You may request to play with a different kart.
        Call `python3 -c "import pystk; pystk.init(pystk.GraphicsConfig.ld()); print(pystk.list_karts())"` to see all values.
    """

    def __init__(self, player_id=0):
        """
        Set up a soccer player.
        The player_id starts at 0 and increases by one for each player added. You can use the player id to figure out
        your team (player_id % 2), or assign different roles to different agents.

        all_players = ['adiumy', 'amanda', 'beastie', 'emule', 'gavroche', 'gnu', 'hexley', 'kiki', 'konqi',
                    'nolok', 'pidgin', 'puffy', 'sara_the_racer', 'sara_the_wizard', 'suzanne', 'tux', 'wilber', 'xue']
        """
        print("starting simulator")
        # select kart
        self.kart = 'xue'
        # set up parameters
        self.state = 'kickoff'
        # self.states = {'kickoff': self.kickoff_action,
        #                 'reset': self.reset_action,
        #                 'attack': self.attack_action,
        #                 'stuck': self.stuck_action
        #                 }
        from collections import deque
        self.past_kart_locs = deque(maxlen=5)
        self.past_puck_locs = deque(maxlen=5)
        self.past_state = deque(maxlen=5)
        self.past_actions = deque(maxlen=5)
        self.state_lock = False
        self.state_lock_turns = 0
        self.current_vel = 0
        self.target_vel = 30
        # load model
        self.model = load_model().eval()

        # Select Team
        self.team = player_id % 2
        # Determine whether player is on offense (position = 0) or defense (position = 1)
        if self.team == 0:
            self.position = (player_id / 2) % 2
        else:
            self.position = (player_id - 1 / 2) % 2

        # assign offense and defense
        if player_id // 2 == 0:
            self.role = 'offense'
        else:
            self.role = 'defense'

    def act(self, image, player_info):
        """
        Set the action given the current image
        :param image: numpy array of shape (300, 400, 3)
        :param player_info: pystk.Player object for the current kart -> https://pystk.readthedocs.io/en/latest/state.html?highlight=pystk.Player#pystk.Player
        return: Dict describing the action
        """
        print('======================================== frame start ======================================')
        action = {'acceleration': 1, 'brake': False, 'drift': False, 'nitro': False, 'rescue': False, 'steer': 0}
        self.current_vel = np.linalg.norm(player_info.kart.velocity)
        if len(self.past_actions) > 0:
            action = self.past_actions[-1]
        # Puck Information
        image_transform = F.to_tensor(image)[None]
        # puck_loc = self.model(image_transform).squeeze(0).detach().cpu().numpy()
        self.image_puck_loc = (self.model(image_transform).detach().cpu().numpy())[0]
        print("Puck image Location", self.image_puck_loc)
        self.puck_loc = self.image_puck_loc

        x = self.image_puck_loc[0]
        y = self.image_puck_loc[1]

        # self.puck_loc = np.float32(self.image_to_local(x, y, player_info))
        # print('puck world loc:', self.puck_loc)


        x_middle = 0
        # Kart Information
        self.kart_loc = self.to_numpy(player_info.kart.location)

        self.kart_front = self.to_numpy(player_info.kart.front)
        print('kart loc', self.kart_loc, self.kart_front)
        # print('kart_angle', np.arctan2([self.kart_loc[1], self.kart_front[1]],[self.kart_loc[0], self.kart_front[0]]) * 180/np.pi)
        # kart_velocity = player_info.kart.velocity
        # kart_attachment = player_info.kart.attachment.type
        # kart_powerup = player_info.kart.powerup.type

        if len(self.past_kart_locs) != 0:
            if self.check_reset(self.kart_loc) == True:
                self.state_lock = False

        # set kart state
        if not self.state_lock:
            self.state = self.set_state(self.kart_loc, self.puck_loc)
        # self.stuck(kart_loc)
        # self.state = 'stuck'
        print('state:', self.state)
        if self.state == 'kickoff':
            # self.puck_loc = (0, 0)
            action = self.kickoff_action(self.kart_loc, self.kart_front, self.puck_loc, action)
        elif self.state == 'in_goal':
            action = self.getOutOfGoal(action)
        elif self.state == 'attack':
            action = self.attack_action(self.kart_loc, self.kart_front, self.puck_loc, action)
        # elif self.state == 'positioning':
        #     action = self.positioning_action(self.kart_loc, self.kart_front, self.puck_loc, action)
        elif self.state == 'searching':
            action = self.searching_action(action)
        elif self.state == 'stuck':
            action = self.stuck_action(self.kart_loc, action)

        else:
            self.state_lock_turns -= 1
            action = self.past_actions[-1]

        if self.state_lock_turns == 0:
            self.state_lock = False

        self.past_kart_locs.append(self.kart_loc)
        self.past_puck_locs.append(self.puck_loc)
        self.past_state.append(self.state)
        self.past_actions.append(action)
        # self.position = (player_id - 1 / 2 ) % 2
        # self.teammate_has_puck = False
        # self.step = 0
        print('action:', action)

        # TODO: Remove puck_loc from the returned before submitting
        return action, self.image_puck_loc


    # ============================================= helper functions =============================================
    # Convert a location to a numpy array
    @staticmethod
    def to_numpy(location):
        """
        Don't care about location[1], which is the height
        """
        return np.float32([location[0], location[2]])

    # Gives you coordinates of x in image based off of the projection and view of something.
    @staticmethod
    def _to_image(x, proj, view):
        W, H = 400, 300
        p = proj @ view @ np.array(list(x) + [1])
        return np.array([W / 2 * (p[0] / p[-1] + 1), H / 2 * (1 - p[1] / p[-1])])

    # Gets vector between two objects given their real world locations.
    @staticmethod
    def get_vector_from_this_to_that(me, obj, normalize=True):
        """
        Expects numpy arrays as input. Normalize returns unit vector.
        """
        vector = obj - me
        if normalize:
            return vector / np.linalg.norm(vector)
        return vector

    def get_unit_vector(vector): # from https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
        """ 
        Returns the unit vector of the vector.  
        """
        return vector / np.linalg.norm(vector)

    def angle_between(self, v1, v2): # from https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
        """ 
        Returns the angle in radians between unit vectors 'v1' and 'v2'::
        >>> angle_between((1, 0, 0), (0, 1, 0))
        1.5707963267948966
        >>> angle_between((1, 0, 0), (1, 0, 0))
        0.0
        >>> angle_between((1, 0, 0), (-1, 0, 0))
        3.141592653589793
        """
        # v1_u = self.get_vector(v1)
        # v2_u = self.get_vector(v2)
        return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))


    def _to_world(self, x, y, proj, view, height=0):
        W, H = 400, 300
        pv_inv = np.linalg.pinv(proj @ view)
        xy, d = pv_inv.dot([float(x) / (W / 2) - 1, 1 -
                            float(y) / (H / 2), 0, 1]), pv_inv[:, 2]
        x0, x1 = xy[:-1] / xy[-1], (xy + d)[:-1] / (xy + d)[-1]
        t = (height - x0[1]) / (x1[1] - x0[1])
        if t < 1e-3 or t > 10:
            # Project the point forward by a certain distance, if it would end up behind
            t = 10
        return t * x1 + (1 - t) * x0

    # Take image coordinates to find real world coordinates based off of player location
    def image_to_local(self, x, y, player):
        y += 40
        proj = np.array(player.camera.projection).T
        view = np.array(player.camera.view).T
        x, _, y = self._to_world(x, y, proj, view, player.kart.location[1])
        # strip ridiculous values
        # if abs(x) > 39 or abs(y) > 64:
        #   return (99, 99)
        # return (x, y)

    # Used to Set the goal locations for each player in the beggining
    def set_goal_loc(self, kart_loc):
        z = kart_loc[-1]
        if z < 0:
            self.our_goal = (0, -64)
            self.their_goal = (0, 64)
        else:
            self.our_goal = (0, 64)
            self.their_goal = (0, -64)

    def puck_lost(self, puck_loc):
        threshold = .5
        checker = 0
        if abs(puck_loc[0]) > threshold or abs(puck_loc[-1]) > threshold:
            checker += 1
        for past_puck in reversed(self.past_puck_locs):
            if abs(past_puck[0]) > threshold or abs(past_puck[-1]) > threshold:
                checker += 1
        if checker / (len(self.past_puck_locs) + 1) > 0.6:
            print('puck lost true', checker)
            return True
        else:
            print('puck lost false', checker)
            return False
    
    def get_orientation(self, kart_loc, puck_loc):
        kart_loc_y = kart_loc[-1]
        kart_front_y = puck_loc[-1]
        if kart_front_y > kart_loc_y:
            print('right is positive')
            return 1
        else:
            print('right is negative')
            return -1
        
    def check_reset(self, kart_loc):
        threshold = 5
        last_loc = self.past_kart_locs[-1]
        x_diff = abs(last_loc[0] - kart_loc[0])
        y_diff = abs(last_loc[-1] - kart_loc[-1])
        if x_diff > threshold or y_diff > threshold:
            print('reset check true', x_diff, y_diff)
            return True
        print('reset check false', x_diff, y_diff)
        return False


    # ============================================= set state logic =============================================
    def set_state(self, kart_loc, puck_loc):
        """
        set current state of the kart
        """
        # set kickoff and start timer for end
        if  self.kickoff(kart_loc) == True:
            self.kickoff_timer = 0
            self.set_goal_loc(kart_loc)
            print('their goal:', self.their_goal, '\nour goal:', self.our_goal)
            return 'kickoff'
        else:
            self.kickoff_timer += 1
    
        if self.state == 'kickoff' and self.kickoff_timer < 30:
            return 'kickoff'
    
        if self.stuck(kart_loc) == True:
            return 'stuck'
        elif self.inGoal(kart_loc) == True:
            return 'in_goal'
        elif self.searching(puck_loc) == True:
            return 'searching'
        # elif self.positioning(kart_loc, puck_loc) == True:
        #     return 'positioning'
        else:
            return 'attack'

    # ============================================= kickoff logic =============================================
    def kickoff(self, kart_loc):
        if len(self.past_kart_locs) == 0:
            print('kickoff check true')
            return True
        return self.check_reset(kart_loc)

    def kickoff_action(self, kart_loc, kart_front, puck_loc, action):
        """
        boost towards center to find puck
        """
        # puck_to_their_goal = np.linalg.norm(puck_loc - np.float32(self.their_goal))
        # print('puck to goal:', puck_to_their_goal)
        action = {'acceleration': 1, 'steer': 4 * puck_loc[0], 'brake': False, 'nitro': True}
        steer_dir = self.get_orientation(kart_loc, kart_front)
        x = kart_loc[0]
        y = kart_loc[-1]
        if x > 3:
            action['steer'] = -.5 * steer_dir
        elif x < -3:
            action['steer'] = .5 * steer_dir
        if abs(y) < 20:
            action['acceleration'] = 0.5
        return action


    # ============================================= in_goal logic ============================================= 
    def inGoal(self, kart_loc):
        if ((kart_loc[1] > 64) or (kart_loc[1] < -64)):
            self.state_lock = True
            self.state_lock_turns = 10
            print('in goal check true')
            return True
        else:
            print('in goal check false')
            return False

    def getOutOfGoal(self, action):
        # In the Blue goal
        if(self.kart_loc[1] > 0):
        # If facing backwards, go backwards
            if (self.kart_front[1] - self.kart_loc[1] > .3):
                action['acceleration'] = 0
                action['brake'] = True
                if (self.kart_loc[0] < 0):
                    action['steer'] = -1
                else:
                    action['steer'] = 1
        # Otherwise you're facing forwards, so accelerate
        else:
            action['acceleration'] = 1
            action['brake'] = False
            if (self.kart_loc[0] < 0):
                action['steer'] = 1
            else:
                action['steer'] = -1
        # In the Red goal
        if (self.kart_loc[1] < 0):
        # If facing backwards, go backwards
            if abs(self.kart_front[1] - self.kart_loc[1]) > .3:
                action['acceleration'] = 0
                action['brake'] = True
            if (self.kart_loc[0] < 0):
                action['steer'] = .2    
            else:
                action['steer'] = -.2
        # Otherwise you're facing forwards, so accelerate
        else:
            action['acceleration'] = 1
            action['brake'] = False
            if (self.kart_loc[0] < 0):
                action['steer'] = -.2
            else:
                action['steer'] = .2

        return action


    # ============================================= stuck logic =============================================
    def stuck(self, kart_loc):
        # print("locations", kart_loc, self.past_kart_locs[-1])
        # print("stuck check Difference", abs(kart_loc - self.past_kart_locs[-1]))
        if ((abs(kart_loc - self.past_kart_locs[-1]) < 0.01).all()):
            self.state_lock = True
            if self.current_vel > 10.0 and self.past_actions[-1]['acceleration'] > 0:
              self.state_lock_turns = 5
            else:
                self.state_lock_turns = 3
            print('stuck check true')
            return True
        else:
            print('kickoff check false')
            return False


    def stuck_action(self, kart_loc, action):
        past_action = self.past_actions[4]
        distance_moved = kart_loc - self.past_kart_locs[0]

        action['brake'] = not past_action['brake']

        if past_action['acceleration'] > 0:
            action['acceleration'] = 0
        else:
            action['acceleration'] = .5

        if(np.sign(distance_moved[0]) == np.sign(distance_moved[1])):
            action['steer'] = 1
        else:
            action['steer'] = -1

        print("stuck action:", action['steer'])
        return action


    # ============================================= positioning logic =============================================
    # def positioning(self, kart_loc, puck_loc):
    #     if self.puck_known_check(puck_loc) == False:
    #         return True
    #     else:
    #         return False

    #     # distance_to_puck = np.linalg.norm(kart_loc - puck_loc)
    #     # kart_to_their_goal = abs(kart_loc[-1] - np.float32(self.their_goal[-1]))
    #     # print('kart to goal dist:', kart_to_their_goal)
    #     # puck_to_their_goal = abs(puck_loc[-1] - np.float32(self.their_goal[-1]))
    #     # print('puck to their goal dist:', puck_to_their_goal)

    #     # if (kart_to_their_goal - puck_to_their_goal) < 0:
    #     #     print('positioning: True')
    #     #     return True
    #     # else:
    #     #     print('positioning: False')
    #     #     return False

    # def positioning_action(self, kart_loc, kart_front, puck_loc, action):
    #     vector_of_kart = self.get_vector_from_this_to_that(kart_loc, kart_front)
    #     vector_to_goal = self.get_vector_from_this_to_that(kart_loc, self.their_goal)
    #     vector_to_puck = self.get_vector_from_this_to_that(kart_loc, puck_loc)

    #     action['brake'] = True
    #     action['acceleration'] = 0
    #     # action['steer'] = -np.sign(np.dot(vector_of_kart, vector_to_puck))
    #     return action


    # ============================================= searching logic =============================================
    def searching(self, puck_loc):
        return self.puck_lost(puck_loc)
        # result = []
        # # Go from 1-3
        # for i in range(1, 4):
        #     # To have -1, -2, -3
        #     negative = -1 * i
        #     last_puck = past_pucks[negative]
        #     last_x = last_puck[0]
        #     last_y = last_puck[1]
        #     current_x = puck_loc[0]
        #     current_y = puck_loc[1]
        #     # May need to change .5
        #     if (abs(last_x - current_x) > .5) and (abs(last_y - current_y) > .5):
        #         result.append(True)
        #     else:
        #         result.append(False)
        # # If all are true
        # if result[0] and result[1] and result[2]:
        #     self.state_lock = True
        #     self.state_lock_turns = 3
        #     return True
        # return False

    def searching_action(self, action):
        # Flip a bitch and zoom
        if self.state_lock_turns == 1:
            action['drift'] = False
            action['steer'] = 0
            action['acceleration'] = 1
        else:
            action['acceleration'] = .5
            action['steer'] = 1
            action['drift'] = True
        return action

    # ============================================= attack logic =============================================
    def attack(self, kart_loc, puck_loc):
        if self.puck_lost(puck_loc) == False:
            return True
        return True


    #     # distance_to_puck = np.linalg.norm(kart_loc - puck_loc)
    #     # kart_to_their_goal = np.linalg.norm(kart_loc - np.float32(self.their_goal))
    #     # puck_to_their_goal = np.linalg.norm(puck_loc - np.float32(self.their_goal))
    #     threshold = 0.1
    #     for past_puck in reversed(self.past_puck_locs):
    #         x_diff = abs(past_puck[0] - puck_loc[0])
    #         y_diff = abs(past_puck[1] - puck_loc[-1])
    #         if x_diff > threshold or y_diff > threshold:
    #             print("check_attack: False")
    #             return False
    #     print("check_attack: True")
    #     return True


    #     # if abs(kart_to_their_goal) > abs(puck_to_their_goal):
    #     #     print('attack: True')
    #     #     return True
    #     # else:
    #     #     print('attack: False')
    #     #     return False

    def attack_action(self, kart_loc, kart_front, puck_loc, action):
        vector_of_kart = self.get_vector_from_this_to_that(kart_loc, kart_front)
        vector_to_goal = self.get_vector_from_this_to_that(kart_loc, self.their_goal)
     
        x = puck_loc[0]
        print('attack x:', x)
        action = {'acceleration': .75, 'steer': x, 'brake': False}

        steer_dir = self.get_orientation(kart_loc, kart_front)
        if x < 0.05 or x > -0.05: # hammer it home!
            print('1')
            action['steer'] = np.sign(np.cross(vector_to_goal, vector_of_kart)) * steer_dir
            action['acceleration'] = 1
            action['nitro'] = True
        elif x > 0.05:
            print('2')
            action['steer'] = .75 * steer_dir
            action['acceleration'] = .25
        elif x < -0.05:
            print('3')
            action['steer'] = -.75 * steer_dir
            action['acceleration'] = .25
        elif x > 0.4:
            print('4')
            action['steer'] = 1 * steer_dir
            action['drift'] = True
            action['acceleration'] = .25
        elif x < -0.4:
            print('5')
            action['steer'] = -1 * steer_dir
            action['drift'] = True
            action['acceleration'] = .25
        
        return action


    # ============================================= defense logic =============================================
    def defense_position(self, kart_loc, puck_loc):
        kart_to_our_goal = np.linalg.norm(kart_loc - np.float32(self.their_goal))
        puck_to_our_goal = np.linalg.norm(puck_loc - np.float32(self.their_goal))

        if abs(kart_to_our_goal) < abs(puck_to_our_goal):
            return True
        else:
            return False

    def defense_action(self, puck_loc, action):
        x = puck_loc[0]
        if x > .3 or x < -.3:
            action['steer'] = np.sign(x) * 1
        else:
            action['steer'] = 0
        if x > .4 or x < -.4:
            action['steer'] = np.sign(x) * 1
            action['drift'] = True
            action['acceleration'] = .5
        else:
            action['drift'] = False
            return action
