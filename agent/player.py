import numpy as np
from .model import PuckDetector, load_model
import torchvision.transforms.functional as F
import torch
import random
import math
from colorama import Fore
from collections import deque

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
        self.player_id = player_id
        self.past_kart_locs = deque(maxlen=5)
        self.past_puck_locs = deque(maxlen=5)
        self.past_state = deque(maxlen=5)
        self.past_actions = deque(maxlen=5)
        # State lock stuff
        self.state_lock = False
        self.state_lock_turns = 0
        self.search_count = 0
        # Velocity values
        self.current_vel = 0
        self.target_vel = 20
        self.last_known_puck = []
        # load model
        self.model = load_model().eval()

        # Select Team
        self.team = player_id % 2
        # Determine whether player is on offense (position = 0) or defense (position = 1)
        if self.team == 0:
            self.position = (player_id / 2) % 2
            self.our_goal_left = (-10, -64)
            self.our_goal_center = (0, -64)
            self.our_goal_right = (10, -64)
            self.their_goal_left = (-10, 64)
            self.their_goal_center = (0, 64)
            self.their_goal_right = (10, 64)
        else:
            self.position = (player_id - 1 / 2) % 2
            self.our_goal_left = (-10, 64)
            self.our_goal_center = (0, 64)
            self.our_goal_right = (10, 64)
            self.their_goal_left = (-10, -64)
            self.their_goal_center = (0, -64)
            self.their_goal_right = (10, -64)

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
        print('Player ', self.player_id)
        action = {'acceleration': 1, 'brake': False, 'drift': False, 'nitro': False, 'rescue': False, 'steer': 0}
        self.current_vel = np.linalg.norm(player_info.kart.velocity)
        if len(self.past_actions) > 0:
            action = self.past_actions[-1]
        # Puck Information
        image_transform = F.to_tensor(image)[None]
        self.image_puck_loc = (self.model(image_transform).detach().cpu().numpy())[0]
        self.puck_loc = self.image_puck_loc

        # add class variables
        self.search_timer = 0
        self.drift_counter = 0

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
            self.state = self.set_state(self.kart_loc, self.kart_front, self.puck_loc)
            # self.stuck(kart_loc)
            # self.state = 'stuck'
            if self.state == 'kickoff':
                # self.puck_loc = (0, 0)
                action = self.kickoff_action(self.kart_loc, self.kart_front, self.puck_loc, action)
            elif self.state == 'in_goal':
                action = self.getOutOfGoal(action)
            elif self.state == 'attack':
                action = self.attack_action(self.kart_loc, self.kart_front, self.puck_loc, action)
                self.last_known_puck = self.image_puck_loc
            elif self.state == 'reposition':
                action = self.reposition_action(self.kart_loc, self.kart_front, self.puck_loc, action)
            elif self.state == 'searching':
                action = self.searching_action(self.kart_loc, self.kart_front, action)
            elif self.state == 'stuck':
                action = self.stuck_action(self.kart_loc, action)

        else:
            action = self.past_actions[-1]
            self.state_lock_turns -= 1
            if self.state_lock_turns == 0:
                self.state_lock = False

        print(Fore.GREEN + 'state: {}'.format(self.state) + Fore.WHITE)

        self.past_kart_locs.append(self.kart_loc)
        self.past_puck_locs.append(self.puck_loc)
        self.past_state.append(self.state)
        self.past_actions.append(action)
        # self.position = (player_id - 1 / 2 ) % 2
        # self.teammate_has_puck = False
        # self.step = 0
        print(Fore.YELLOW + 'action: {}'.format(action) + Fore.WHITE)
        print("Current Speed", self.current_vel)

        # TODO: Remove puck_loc from the returned before submitting
        return action, self.image_puck_loc


    # ============================================= set state logic =============================================
    def set_state(self, kart_loc, kart_front, puck_loc):
        """
        set current state of the kart
        """
        # set kickoff and start timer for end
        if  self.kickoff(kart_loc) == True:
            self.kickoff_timer = 0
            # self.set_goal_loc(kart_loc)
            print('their goal:', self.their_goal_left, self.their_goal_right ,'\nour goal:', self.our_goal_left, self.our_goal_right)
            return 'kickoff'
        else:
            self.kickoff_timer += 1
    
        if self.state == 'kickoff' and self.kickoff_timer < 33:
            return 'kickoff'

        if self.stuck(kart_loc) == True:
            return 'stuck'
        elif self.inGoal(kart_loc) == True:
            return 'in_goal'
        elif self.reposition(kart_loc, kart_front, puck_loc) == True:
            return 'reposition'
        elif self.searching(puck_loc) == True:
            self.search_timer += 1
            return 'searching'
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
        # steer_dir = self.get_orientation()
        x = kart_loc[0]
        y = kart_loc[-1]
        if x > 3:
            action['steer'] = -.35
        elif x < -3:
            action['steer'] = .35
        # if abs(y) < 20 and x < .05 and x > -.05:
        #     action['nitro'] = True

        # if self.role == 'defense':
        #     action = {'acceleration': .25, 'steer': 0, 'brake': False, 'nitro': False}
        #     if x > 3:
        #         action['steer'] = -.4
        #     elif x < -3:
        #         action['steer'] = .4
        #     if abs(y) < 55:
        #         action['acceleration'] = 0

        return action


    # ============================================= in_goal logic ============================================= 
    def inGoal(self, kart_loc):
        if ((kart_loc[1] > 66) or (kart_loc[1] < -66)):
            self.state_lock = True
            self.state_lock_turns = 10
            print('in goal check true')
            return True
        else:
            print('in goal check false')
            return False

    def getOutOfGoal(self, action):
        goal = 64 * np.sign(self.kart_loc[1])
        # In the Blue goal
        if(self.kart_loc[1] > 0):
        # If facing backwards, go backwards
            if (self.kart_front[1] - self.kart_loc[1] > -.3):
                action['acceleration'] = 0
                action['brake'] = True
                if self.last_known_puck[0] < self.kart_loc[0]:
                    action['steer'] = 1
                else:
                    action['steer'] = -1
            # Otherwise you're facing forwards, so accelerate
            else:
                action['acceleration'] = 1
                action['brake'] = False
                if self.last_known_puck[0] > self.kart_loc[0]:
                    action['steer'] = -1
                else:
                    action['steer'] = 1

          # In the Red goal
        else:
            # If facing backwards, go backwards
            if abs(self.kart_front[1] - self.kart_loc[1]) > .3:
                action['acceleration'] = 0
                action['brake'] = True
                if self.last_known_puck[0] < self.kart_loc[0]:
                    action['steer'] = -1
                else:
                    action['steer'] = 1
            # Otherwise you're facing forwards, so accelerate
            else:
                action['acceleration'] = 1
                action['brake'] = False
                if self.last_known_puck[0] < self.kart_loc[0]:
                    action['steer'] = 1
                else:
                    action['steer'] = -1
        if abs(self.kart_loc[1]) > 69:
            action['steer'] = action['steer'] * ((10 - abs(self.kart_loc[0])) / 10)
        action['nitro'] = False
        return action


    # ============================================= stuck logic =============================================
    # You are stuck if:
    # 1. You do not move within the given threshold AND
    # 2. You are moving under a certain velocity threshold
    # 3. You are not trying to move (in case you' crashed into another car)
    # 4. 1 AND 2 within specific geographic locations X ( -45, 45) and/or Y (- 63.5, 63.5)
    def stuck(self, kart_loc):
        print("locations", kart_loc, self.past_kart_locs[-1])
        print("Difference", abs(kart_loc - self.past_kart_locs[-1]))
        print("NUMPY ", (abs(kart_loc - self.past_kart_locs[-1]) < 0.05).all())
        no_move = (abs(self.kart_loc - self.past_kart_locs[-1]) < 0.01).all()
        no_vel = self.current_vel < 2.0
        no_try_move = (self.past_actions[0]['brake'] == False and self.past_actions[0]['acceleration'] != 0)
        danger_zone = abs(self.kart_loc[0]) >= 45 or abs(self.kart_loc[1]) >= 63.5
        if no_move and no_vel and (no_try_move or danger_zone):
            self.state_lock = True
            if self.current_vel > 10.0 and self.past_actions[-1]['acceleration'] > 0:
              self.state_lock_turns = 10
            else:
                self.state_lock_turns = 7
            print('stuck check true')
            return True
        else:
            print('kickoff check false')
            return False


    def stuck_action(self, kart_loc, action):
        past_action = self.past_actions[4]
        distance_moved = kart_loc - self.past_kart_locs[0]

        # action['brake'] = not past_action['brake']


        # if past_action['acceleration'] > 0:
        #     action['acceleration'] = 0
        # else:
        #     action['acceleration'] = .5

        if (self.kart_loc[1] < 0):
            print("blue goal side")
            if (self.kart_front[1] - self.kart_loc[1] < 0):
                print("kart facing goal")
                action['acceleration'] = 0
                action['brake'] = True
            else:
                print("kart facing red goal")
                action['acceleration'] = 1
        else:
            print("red goal side")
            if (self.kart_front[1] - self.kart_loc[1] > -0.001):
                print("kart facing goal")
                action['acceleration'] = 0
                action['brake'] = True
            else:
                print("kart facing blue goal")
                action['acceleration'] = 1

        if (abs(self.kart_loc[0]) >= 45):
            if (action['acceleration'] > 0 ):
                action['steer'] = np.sign(self.kart_loc[0]) * -1
            else:
                action['steer'] = np.sign(self.kart_loc[0]) * 1
        else:
            if (self.last_known_puck[1] > self.kart_loc[1]):

                if(self.kart_loc[0] < 0):
                    print("does this proc")
                    action['steer'] = 1
                else:
                    ###print("does this proc??")
                    action['steer'] = -1

            elif(self.last_known_puck[1] < self.kart_loc[1]):
                if(self.kart_loc[0] < 0):
                    action['steer'] = -1
                else:
                    action['steer'] = 1
        # if(np.sign(distance_moved[0]) == np.sign(distance_moved[1])):
        #     action['steer'] = 1
        # else:
        #     action['steer'] = -1

        action['nitro'] = False
        ###print("stuck action:", action)
        return action


    # ============================================= positioning logic =============================================
    def reposition(self, kart_loc, kart_front, puck_loc):
        # threshold = -.8
        # print("SEARCH CHECK", puck_loc)
        # if puck_loc[1] < threshold and self.search_count < 3:
        #     self.search_count += 1
        #     print("COUNTTTTTT", self.search_count)
        # elif self.search_count > 0:
        #     self.search_count -= 1

        # if self.search_count == 3:
        #     return True
        # elif self.search_count == 2 and self.state == 'searching':
        #     return True

        return False
        # vector_of_kart = self.get_vector_from_this_to_that(kart_loc, kart_front)
        # vector_to_their_goal = self.get_vector_from_this_to_that(kart_loc, self.their_goal_center)
        # a = self.angle_between(vector_of_kart, vector_to_their_goal)

        # if math.degrees(a) > 90:
        #     self.state_lock = True
        #     self.state_lock_turns = 10
        #     print('reposition true')
        #     return True
        # else:
        #     print('reposition false')
            # return False


            # drive towards our goal and turn around
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

    def reposition_action(self, kart_loc, kart_front, puck_loc, action):
        action['acceleration'] = 0
        action['brake'] = True

        return action
        # vector_of_kart = self.get_vector_from_this_to_that(kart_loc, kart_front)
        # vector_to_their_goal = self.get_vector_from_this_to_that(kart_loc, self.their_goal_center)
        # vector_to_our_goal = self.get_vector_from_this_to_that(kart_loc, self.our_goal_center)
        # dist_to_our_goal = np.linalg.norm(kart_loc, self.our_goal_center)
        
        # action['steer'] = np.sign(np.dot(vector_of_kart, vector_to_our_goal))
        # action['acceleration'] = 1

        # if dist_to_our_goal < 10:
        #     action['steer'] = np.sign(np.dot(vector_of_kart, vector_to_their_goal))
        #     action['acceleration'] = 0
            
    #     vector_of_kart = self.get_vector_from_this_to_that(kart_loc, kart_front)
    #     vector_to_goal = self.get_vector_from_this_to_that(kart_loc, self.their_goal)
    #     vector_to_puck = self.get_vector_from_this_to_that(kart_loc, puck_loc)

    #     action['brake'] = True
    #     action['acceleration'] = 0
    #     # action['steer'] = -np.sign(np.dot(vector_of_kart, vector_to_puck))
    #     return action


    # ============================================= searching logic =============================================
    # Basically, this function will check if puck location is presumed to be "off the map"
    # If it finds it, we require threshold to be broken 3 times in a row b4 toggle searching mode
    # Then wait until you lock into an "actual" location two times in a row before getting out of
    # search mode.
    def searching(self, puck_loc):
        threshold = -.8
        print("SEARCH CHECK", puck_loc)
        if puck_loc[1] < threshold and self.search_count < 3:
            self.search_count += 1
            print("COUNTTTTTT", self.search_count)
        elif self.search_count > 0:
            self.search_count -= 1

        if self.search_count == 3:
            self.state_lock = True
            self.state_lock_turns = 7
            return True
        elif self.search_count == 2 and self.state == 'searching':
            self.state_lock = True
            self.state_lock_turns = 7
            return True

        return False
        # if puck_loc[0] < threshold and puck_loc[-1] < threshold:
        # for past_puck in self.past_puck_locs[0:1]:
        #     if past_puck < 0.8:
        #         checker+=1
        # if checker =
        # for past_puck in reversed(self.past_puck_locs):
        #     if past_puck[0] < threshold and past_puck[-1] < threshold:
        #         checker += 1
        # if checker / (len(self.past_puck_locs) + 1) >= 0.8:
        #     print('puck lost true', checker)
        #     return True
        # else:
        #     print('puck lost false', checker)
        #     return False

    # def searching_action(self, action):
    #     result = []
    #     # Go from 1-3
    #     for i in range(1, 4):
    #         # To have -1, -2, -3
    #         negative = -1 * i
    #         last_puck = self.past_puck_locs[negative]
    #         last_x = last_puck[0]
    #         last_y = last_puck[1]
    #         current_x = puck_loc[0]
    #         current_y = puck_loc[1]
    #         # May need to change .5
    #         if (abs(last_x - current_x) > .5) or (abs(last_y - current_y) > .5):
    #             result.append(True)
    #         else:
    #             result.append(False)
    #     # If all are true
    #     if result[0] and result[1] and result[2]:
    #         return True
    #     return False

    def searching_action(self, kart_loc, kart_front, action):
        
        if self.search_timer < 15:
            print
            self.search_timer += 1
            action['steer'] = 0
            action['acceleration'] = 0
            action['brake'] = True
        else:
            # Flip a bitch and zoom
            # Check quadrants
            kart_x = kart_loc[0]
            kart_y = kart_loc[1]
            front_x = kart_front[0]
            front_y = kart_front[1]\
            # bottom-left
            if kart_x < 0 and kart_y < 0:
                if front_x > kart_x and front_y > kart_y:
                    action['acceleration'] = 1
                    action['steer'] = 0
                    action['drift'] = False
                # Facing negative
                elif front_x < kart_x:
                    action['steer'] = 1
                    action['drift'] = True
                    action['acceleration'] = .3
                # Facing towards wall
                else:
                    action['steer'] = -1
                    action['drift'] = True
                    action['acceleration'] = .3
            # top-left
            elif kart_x < 0:
                if front_x > kart_x and front_y < kart_y:
                    action['acceleration'] = 1
                    action['steer'] = 0
                    action['drift'] = False
                # Facing negative
                elif front_x < kart_x:
                    action['steer'] = -1
                    action['drift'] = True
                    action['acceleration'] = .3
                # Facing towards wall
                else:
                    action['steer'] = 1
                    action['drift'] = True
                    action['acceleration'] = .3
            # bottom-right
            elif kart_x > 0 and kart_y < 0:
                if front_x < kart_x and front_y > kart_y:
                    action['acceleration'] = 1
                    action['steer'] = 0
                    action['drift'] = False
                # Facing positive
                elif front_x > kart_x:
                    action['steer'] = -1
                    action['drift'] = True
                    action['acceleration'] = .3
                # Facing towards wall
                else:
                    action['steer'] = 1
                    action['drift'] = True
                    action['acceleration'] = .3
            # top-right
            else:
                if front_x < kart_x and front_y < kart_y:
                    action['acceleration'] = 1
                    action['steer'] = 0
                    action['drift'] = False
                # Facing positive
                elif front_x > kart_x:
                    action['steer'] = 1
                    action['drift'] = True
                    action['acceleration'] = .3
                # Facing towards wall
                else:
                    action['steer'] = -1
                    action['drift'] = True
                    action['acceleration'] = .3

        return action



    # def searching_action(self, kart_loc, kart_front, action):
    #     # Flip a bitch and zoom
    #     # Check quadrants
    #     kart_x = kart_loc[0]
    #     kart_y = kart_loc[1]
    #     # bottom-left
    #     if kart_x < 0 and kart_y < 0:
    #         facing = (kart_front[0] - kart_loc[0]) > 0
    #         # Facing positive
    #         if facing:
    #             action['steer'] = .3
    #             action['drift'] = False
    #             action['acceleration'] = 1
    #         # Facing negative
    #         else:
    #             action['steer'] = 1
    #             action['drift'] = True
    #             action['acceleration'] = .5
    #     # top-left
    #     elif kart_x < 0:
    #         facing = (kart_front[0] - kart_loc[0]) > 0
    #         # Facing positive
    #         if facing:
    #             action['steer'] = -.3
    #             action['drift'] = False
    #             action['acceleration'] = 1
    #         # Facing negative
    #         else:
    #             action['steer'] = -1
    #             action['drift'] = True
    #             action['acceleration'] = .5
    #     # bottom-right
    #     elif kart_x > 0 and kart_y < 0:
    #         facing = (kart_front[0] - kart_loc[0]) > 0
    #         # Facing positive
    #         if facing:
    #             action['steer'] = -1
    #             action['drift'] = True
    #             action['acceleration'] = .5
    #         # Facing negative
    #         else:
    #             action['steer'] = -.3
    #             action['drift'] = False
    #             action['acceleration'] = 1
    #     # top-right
    #     else:
    #         facing = (kart_front[0] - kart_loc[0]) > 0
    #         # Facing positive
    #         if facing:
    #             action['steer'] = 1
    #             action['drift'] = True
    #             action['acceleration'] = .5
    #         # Facing negative
    #         else:
    #             action['steer'] = .3
    #             action['drift'] = False
    #             action['acceleration'] = 1
    #     return action

    def attack_action(self, kart_loc, kart_front, puck_loc, action):
        vector_of_kart = self.get_vector_from_this_to_that(kart_loc, kart_front)
        vector_to_their_goal = self.get_vector_from_this_to_that(kart_loc, self.their_goal_center)
        facing_angle = math.degrees(self.angle_between(vector_of_kart, vector_to_their_goal))

        vector_right = self.get_vector_from_this_to_that(kart_loc, self.their_goal_right)
        vector_center = self.get_vector_from_this_to_that(kart_loc, self.their_goal_center)
        vector_left = self.get_vector_from_this_to_that(kart_loc, self.their_goal_left)
        attack_cone = math.degrees(self.angle_between(vector_left, vector_right))


        # apply smoothing to reduce effects of erratic puck detections
        # x_smoother = []
        # x_smoother.append(puck_loc[0])
        # for past_puck in reversed(self.past_puck_locs):
        #     x_smoother.append(past_puck[0])
        # x = np.median(x_smoother)

        x = puck_loc[0]
        y = puck_loc[-1]

        # if np.sign(vector_center[0]) == np.sign(vector_of_kart[0])

        

        # past_x = self.past_puck_locs[-1][0]
        # if abs(x - past_x) > .5:
        #     x = past_x
        print('attack x:', x)

        action = {'acceleration': .5, 'steer': x, 'brake': False, 'drift': False, 'nitro': False}
        
        if x < 0.05 and x > -0.05:
            print('1')
            # action['steer'] = np.sign(np.cross(vector_to_goal, vector_of_kart))
            action['steer'] = x 
            action['acceleration'] = .5
        elif x > 0.05 and x < .2:
            print('2')
            action['steer'] = .75
            # action['steer'] = -2 * np.sign(np.cross(vector_to_goal, vector_of_kart))
            action['acceleration'] = .4
            action['brake'] = True
        elif x < -0.05 and x > -.3:
            print('3')
            action['steer'] = -.9
            # action['steer'] = -2 * np.sign(np.cross(vector_to_goal, vector_of_kart))
            action['acceleration'] = .4
            action['brake'] = True
        elif x > .2 and x < .4:
            print('2')
            action['steer'] = .9
            # action['steer'] = -2 * np.sign(np.cross(vector_to_goal, vector_of_kart))
            action['acceleration'] = .2
            action['brake'] = True
            action['drift'] = True
        elif x < -.2 and x > -.4:
            print('3')
            action['steer'] = -.75
            # action['steer'] = -2 * np.sign(np.cross(vector_to_goal, vector_of_kart))
            action['acceleration'] = .2
            action['brake'] = True
            action['drift'] = True
        elif x > 0.4 and x < 0.7:
            print('4')
            action['steer'] = 1
            # action['steer'] = 4 * np.sign(np.cross(vector_to_goal, vector_of_kart))
            action['acceleration'] = .1
            action['brake'] = True
            action['drift'] = True
        elif x < -0.4 and x > -0.7:
            print('5')
            action['steer'] = -1 
            # action['steer'] = 4 * np.sign(np.cross(vector_to_goal, vector_of_kart))
            action['acceleration'] = .1
            action['brake'] = True
            action['drift'] = True
        elif x > 0.8:
            print('4')
            action['steer'] = 1
            # if np.sign(self.current_vel) == -1:
            #     action['steer'] = -1
            # action['steer'] = 4 * np.sign(np.cross(vector_to_goal, vector_of_kart))
            action['acceleration'] = .4
            action['brake'] = True
            action['drift'] = False
        elif x < -0.8:
            print('5')
            action['steer'] = -1
            # if np.sign(self.current_vel) == -1:
            #     action['steer'] = 1
            # action['steer'] = 4 * np.sign(np.cross(vector_to_goal, vector_of_kart))
            action['acceleration'] = .4
            action['brake'] = True
            action['drift'] = True
            


        # check if we just exited search to slow kart
        if self.past_state[-1] == 'searching': 
            action['acceleration'] = .5
            action['drift'] = False
            action['nitro'] = False

        
        # check for cross-motion to head off puck
        # if self.cross_motion(puck_loc, self.past_puck_locs, action) == True: 
        #     self.drift_counter += 1
        #     if self.drift_counter < 5:
        #         if puck_loc[0] < self.past_puck_locs[-1][0]:
        #             action['steer'] = -1
        #             # action['drift'] = True
        #             action['nitro'] = False

        #         else:
        #             action['steer'] = 1
        #             # action['drift'] = True
        #             action['nitro'] = False
        # else:
        #     self.drift_counter = 0

        if abs(x) < .2 and abs(y) < .2:
            print(Fore.MAGENTA + 'hammer time!' + Fore.WHITE)
            if facing_angle < attack_cone / 2: # within attack cone - hammer it home!
                action['steer'] = 0
                action['acceleration'] = 1
            else: # steer towards attack cone
                action['steer'] = np.sign(vector_center[0])
                action['acceleration'] = .5


        if np.sign(x) != np.sign(action['steer']):
            print(Fore.RED + 'error: turned wrong way - x: {}, steer: {}'.format(x, action['steer']) + Fore.WHITE)

        return action

    # ============================================= defense logic =============================================
    def defense_position(self, kart_loc, puck_loc):
        kart_to_our_goal = np.linalg.norm(kart_loc - np.float32((0 , self.their_goal_left[1])))
        puck_to_our_goal = np.linalg.norm(puck_loc - np.float32((0 , self.their_goal_left[1])))

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

    def get_unit_vector(
      vector):  # from https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
        """
        Returns the unit vector of the vector.
        """
        return vector / np.linalg.norm(vector)

    def angle_between(self, v1,
                      v2):  # from https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
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

    def cross_motion(self, puck_loc, past_puck_locs, action):
        # steering = action['steer']
        # if abs(steering) < .2:
        cross_motion = []
        # vertical_motion = []
        for past_puck in reversed(past_puck_locs):
            x = abs(puck_loc[0] - past_puck[0])
            # y = abs(puck_loc[-1] - past_puck[-1])
            cross_motion.append(x)
            # vertical_motion.append(y)
        avg_x = np.mean(cross_motion)
        # avg_y = np.mean(vertical_motion)
        # movement = []
        if avg_x > .1:
            print('cross motion true')
            return True
            # movement.append(avg_x)
        # if avg_y > .1:
        #     movement.append(avg_y)
        print('cross motion false')
        return False

    # def get_orientation(self):
    #     kart_loc_x = self.kart_loc[0]
    #     kart_loc_y = self.kart_loc[-1]
    #     kart_front_x = self.kart_front[0]
    #     kart_front_y = self.kart_front[-1]
        
    #     # facing our goal
    #     if abs(kart_front_y) > abs(kart_loc_y) and np.sign(self.our_goal_center[-1]) == np.sign(kart_loc_y):
    #         if abs(kart_front_x) > abs(kart_loc_x): #facing out
    #             # turn left
    #             action['steer'] = -1
    #         else: # facing in
    #             # turn 
    #             action['steer'] = 1

    #     # facing their goal
    #     elif abs(kart_front_y) > abs(kart_loc_y) and np.sign(self.our_goal_center[-1]) == np.sign(kart_loc_y):
    #         if abs(kart_front_x) > abs(kart_loc_x): #facing out
    #             # turn left
    #             action['steer'] = -1
    #         else: # facing in
    #             # turn right
    #             action['steer'] = 1
        
    #     # facing center line
    #     elif abs(kart_front_y) < abs(kart_loc_y):
    

    def check_reset(self, kart_loc):
        threshold = 5
        last_loc = self.past_kart_locs[-1]
        x_diff = abs(last_loc[0] - kart_loc[0])
        y_diff = abs(last_loc[-1] - kart_loc[-1])
        if x_diff > threshold or y_diff > threshold:
            # print('reset check true', x_diff, y_diff)
            return True
        # print('reset check false', x_diff, y_diff)
        return False
