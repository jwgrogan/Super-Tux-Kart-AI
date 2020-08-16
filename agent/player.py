import numpy as np
from .model import PuckDetector, load_model
import torchvision.transforms.functional as F
import math
from collections import deque
# from colorama import Fore
class HockeyPlayer:
    """
        Your ice hockey player. You may do whatever you want here. There are three rules:
        1. no calls to the pystk library (your code will not run on the tournament system if you do)
        2. There needs to be a deep network somewhere in the loop
        3. You code must run in 100 ms / frame on a standard desktop CPU (no for testing GPU)
        Try to minimize library dependencies, nothing that does not install through pip on linux.
    """

    def __init__(self, player_id=0):
        # select kart
        self.kart = 'xue'
        # set up parameters
        self.state = 'kickoff'

        self.player_id = player_id
        self.past_kart_locs = deque(maxlen=5)
        self.past_puck_locs = deque(maxlen=5)
        self.past_state = deque(maxlen=5)
        self.past_actions = deque(maxlen=5)
        # State lock stuff
        self.state_lock = False
        self.state_lock_turns = 0
        self.stuck_count = 0
        self.search_count = 0
        # Velocity values
        self.current_vel = 0
        self.target_vel = 25
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
        #print('======================================== frame start ======================================')
        #print('Player ', self.player_id)
        action = {'acceleration': 1, 'brake': False, 'drift': False, 'nitro': False, 'rescue': False, 'steer': 0}
        self.current_vel = np.linalg.norm(player_info.kart.velocity)
        acc_mult = 1
        if len(self.past_actions) > 0:
            action = self.past_actions[-1]
        # Puck Information
        image_transform = F.to_tensor(image)[None]
        self.image_puck_loc = (self.model(image_transform).detach().cpu().numpy())[0]
        self.puck_loc = self.image_puck_loc
        #print("PUCK LOCATIONSsssss", self.puck_loc)
        # add class variables
        self.drift_counter = 0

        # Kart Information
        self.kart_loc = self.to_numpy(player_info.kart.location)

        self.kart_front = self.to_numpy(player_info.kart.front)
        #print('kart loc', self.kart_loc, self.kart_front)

        if len(self.past_kart_locs) != 0:
            if self.check_reset(self.kart_loc) == True:
                self.state_lock = False

        # set kart state
        if not self.state_lock:
            self.state = self.set_state(self.kart_loc, self.kart_front, self.puck_loc)
            if self.state == 'kickoff':
                action = self.kickoff_action(self.puck_loc)
            elif self.state == 'in_goal':
                action = self.getOutOfGoal(action)
            elif self.state == 'attack':
                action = self.attack_action(self.kart_loc, self.kart_front, self.puck_loc, action)
                self.last_known_puck = self.image_puck_loc
            elif self.state == 'searching':
                action = self.searching_action(self.kart_loc, self.kart_front, action)
            elif self.state == 'stuck':
                action = self.stuck_action(self.kart_loc, action)

        else:
            action = self.past_actions[-1]
            self.state_lock_turns -= 1
            if self.state_lock_turns == 0:
                self.state_lock = False

        #print(Fore.GREEN + 'state: {}'.format(self.state) + Fore.WHITE)

        self.past_kart_locs.append(self.kart_loc)
        self.past_puck_locs.append(self.puck_loc)
        self.past_state.append(self.state)
        self.past_actions.append(action)

        if (action['acceleration'] > 0):
            action['brake'] = False

        #Correct acceleration if needed:
        ratio = self.current_vel / self.target_vel

        if self.state != 'kickoff':
            if ratio <= 0.5:
                acc_mult = 1
            elif ratio <= 0.7:
                acc_mult = 0.7
            else:
                acc_mult = 0

        action['acceleration'] *= acc_mult
        #print(Fore.YELLOW + 'action: {}'.format(action) + Fore.WHITE)
        #print("Current Speed", self.current_vel)

        return action


    # ============================================= set state logic =============================================
    def set_state(self, kart_loc, kart_front, puck_loc):
        """
        set current state of the kart
        """
        # set kickoff and start timer for end
        if  self.kickoff(kart_loc) == True:
            self.kickoff_timer = 0
            #print('their goal:', self.their_goal_left, self.their_goal_right ,'\nour goal:', self.our_goal_left, self.our_goal_right)
            return 'kickoff'
        else:
            self.kickoff_timer += 1
    
        if self.state == 'kickoff' and self.kickoff_timer < 33:
            return 'kickoff'

        if self.stuck(kart_loc) == True:
            return 'stuck'
        elif self.inGoal(kart_loc) == True:
            return 'in_goal'
        elif self.searching(puck_loc) == True:
            return 'searching'
        else:
            return 'attack'

    # ============================================= kickoff logic =============================================
    def kickoff(self, kart_loc):
        if len(self.past_kart_locs) == 0:
            #print('kickoff check true')
            return True
        return self.check_reset(kart_loc)

    def kickoff_action(self, puck_loc):
        """
        boost towards center to find puck
        """
        action = {'acceleration': 1, 'steer': 4 * puck_loc[0], 'brake': False, 'nitro': True}
        x = self.kart_loc[0]
        y = self.kart_loc[-1]
        if x > 3:
            action['steer'] = -.32
        elif x < -3:
            action['steer'] = .32

        return action

    # ============================================= in_goal logic ============================================= 
    def inGoal(self, kart_loc):
        if abs(kart_loc[0]) < 10 and ((kart_loc[1] > 63.8) or (kart_loc[1] < -63.8)):
            self.state_lock = True
            self.state_lock_turns = 10
            #print('in goal check true')
            return True
        else:
            #print('in goal check false')
            return False

    def getOutOfGoal(self, action):
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
            if self.kart_front[1] - self.kart_loc[1] < .3:
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
        #print("locations", kart_loc, self.past_kart_locs[-1])
        #print("Difference", abs(kart_loc - self.past_kart_locs[-1]))
        #print("NUMPY ", (abs(kart_loc - self.past_kart_locs[-1]) < 0.05).all())
        no_move = (abs(self.kart_loc - self.past_kart_locs[-1]) < 0.02).all()
        no_vel = self.current_vel < 2.0
        no_try_move = (self.past_actions[0]['brake'] == False and self.past_actions[0]['acceleration'] != 0)
        danger_zone = abs(self.kart_loc[0]) >= 45 or abs(self.kart_loc[1]) >= 63.5
        #print("stuck mode", no_move, no_vel, no_try_move, danger_zone)
        if no_move and no_vel and no_try_move:
            if self.stuck_count < 5:
                #print("hello the weird place,", self.stuck_count)
                self.stuck_count +=1
            else:
                # Reset stuck count because we're triggering stuck
                #print(" the weird place")
                self.stuck_count = 0
                self.state_lock = True
                if self.current_vel > 10.0 and self.past_actions[-1]['acceleration'] > 0:
                  self.state_lock_turns = 10
                else:
                    self.state_lock_turns = 7
                return True

        if no_move and no_vel and (danger_zone):
            self.state_lock = True
            if self.current_vel > 10.0 and self.past_actions[-1]['acceleration'] > 0:
              self.state_lock_turns = 10
            else:
                self.state_lock_turns = 7
            #print('stuck check true')
            return True
        else:
            #print('kickoff check false')
            return False

    def stuck_action(self, kart_loc, action):
        if (self.kart_loc[1] < 0):
            #print("blue goal side")
            if (self.kart_front[1] - self.kart_loc[1] < 0):
                #print("kart facing goal")
                action['acceleration'] = 0
                action['brake'] = True
            else:
                #print("kart facing red goal")
                action['acceleration'] = 1
        else:
            #print("red goal side")
            if (self.kart_front[1] - self.kart_loc[1] > -0.001):
                #print("kart facing goal")
                action['acceleration'] = 0
                action['brake'] = True
            else:
                #print("kart facing blue goal")
                action['acceleration'] = 1

        if (abs(self.kart_loc[0]) >= 45):
            if (action['acceleration'] > 0 ):
                action['steer'] = np.sign(self.kart_loc[0]) * -1
            else:
                action['steer'] = np.sign(self.kart_loc[0]) * 1
        else:
            if (self.last_known_puck[1] > self.kart_loc[1]):

                if(self.kart_loc[0] < 0):
                    #print("does this proc")
                    action['steer'] = 1
                else:
                    ####print("does this proc??")
                    action['steer'] = -1

            elif(self.last_known_puck[1] < self.kart_loc[1]):
                if(self.kart_loc[0] < 0):
                    action['steer'] = -1
                else:
                    action['steer'] = 1

        action['nitro'] = False
        return action

    # ============================================= searching logic =============================================
    # Basically, this function will check if puck location is presumed to be "off the map"
    # If it finds it, we require threshold to be broken 3 times in a row b4 toggle searching mode
    # Then wait until you lock into an "actual" location two times in a row before getting out of
    # search mode.
    def searching(self, puck_loc):
        threshold = -.8
        if puck_loc[1] < threshold and self.search_count < 3:
            self.search_count += 1
        elif self.search_count > 0:
            self.search_count -= 1

        if self.search_count == 3:
            return True
        elif self.search_count == 2 and self.state == 'searching':
            return True

        return False

    def searching_action(self, kart_loc, kart_front, action):
        # Flip a bitch and zoom
        # Check quadrants
        kart_x = kart_loc[0]
        kart_y = kart_loc[1]
        perspective = np.sign(kart_front[0] - kart_loc[0])
        # bottom-left
        if kart_x < 0 and kart_y < 0:
            facing = (kart_front[1] - kart_loc[1]) > 0
            #print("bot left", facing)
            # Facing positive
            if facing:
                action['steer'] = .5
                action['drift'] = True
                action['acceleration'] = 1
            # Facing negative
            else:
                action['steer'] = -.5
                action['drift'] = True
                action['acceleration'] = .5
        # top-left
        elif kart_x < 0:
            facing = (kart_front[1] - kart_loc[1]) > 0
            #print("top left", facing)
            # Facing positive
            if facing:
                action['steer'] = .5
                action['drift'] = True
                action['acceleration'] = 1
            # Facing negative
            else:
                action['steer'] = -.5
                action['drift'] = True
                action['acceleration'] = .5
        # bottom-right
        elif kart_x > 0 and kart_y < 0:
            facing = (kart_front[1] - kart_loc[1]) > 0
            #print("bot right", facing)
            # Facing positive
            if facing:
                action['steer'] = -.5
                action['drift'] = True
                action['acceleration'] = .5
            # Facing negative
            else:
                action['steer'] = .5
                action['drift'] = True
                action['acceleration'] = 1
        # top-right
        else:
            facing = (kart_front[1] - kart_loc[1]) > 0
            #print("top right", facing)
            # Facing positive
            if facing:
                action['steer'] = -.5
                action['drift'] = True
                action['acceleration'] = .5
            # Facing negative
            else:
                action['steer'] = .5
                action['drift'] = True
                action['acceleration'] = 1
        if abs(self.kart_loc[0]) < 20:
            action['steer'] = perspective * action['steer']
        return action

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

        #print('attack x:', x)
        #print("PUCK LOC", puck_loc)
        action = {'acceleration': .5, 'steer': x, 'brake': False, 'drift': False, 'nitro': False}
        
        if x < 0.05 and x > -0.05:
            #print('1')
            # action['steer'] = np.sign(np.cross(vector_to_goal, vector_of_kart))
            action['steer'] = x 
            action['acceleration'] = .5
        elif x > 0.05 and x < .2:
            #print('2')
            action['steer'] = .25
            # action['steer'] = -2 * np.sign(np.cross(vector_to_goal, vector_of_kart))
            action['acceleration'] = .4
            action['brake'] = True
        elif x < -0.05 and x > -.2:
            #print('3')
            action['steer'] = -.25
            # action['steer'] = -2 * np.sign(np.cross(vector_to_goal, vector_of_kart))
            action['acceleration'] = .4
            action['brake'] = True
        elif x > .2 and x < .4:
            #print('2')
            action['steer'] = .75
            # action['steer'] = -2 * np.sign(np.cross(vector_to_goal, vector_of_kart))
            action['acceleration'] = .2
            action['brake'] = True
            action['drift'] = True
        elif x < -.2 and x > -.4:
            #print('3')
            action['steer'] = -.75
            # action['steer'] = -2 * np.sign(np.cross(vector_to_goal, vector_of_kart))
            action['acceleration'] = .2
            action['brake'] = True
            action['drift'] = True
        elif x > 0.4 and x < 0.7:
            #print('4')
            action['steer'] = 1
            # action['steer'] = 4 * np.sign(np.cross(vector_to_goal, vector_of_kart))
            action['acceleration'] = .1
            action['brake'] = True
            action['drift'] = True
        elif x < -0.4 and x > -0.7:
            #print('5')
            action['steer'] = -1 
            # action['steer'] = 4 * np.sign(np.cross(vector_to_goal, vector_of_kart))
            action['acceleration'] = .1
            action['brake'] = True
            action['drift'] = True
        elif x > 0.7:
            #print('4')
            action['steer'] = 1
            # if np.sign(self.current_vel) == -1:
            #     action['steer'] = -1
            # action['steer'] = 4 * np.sign(np.cross(vector_to_goal, vector_of_kart))
            action['acceleration'] = .3
            action['brake'] = True
            action['drift'] = False
        elif x < -0.7:
            #print('5')
            action['steer'] = -1
            # if np.sign(self.current_vel) == -1:
            #     action['steer'] = 1
            # action['steer'] = 4 * np.sign(np.cross(vector_to_goal, vector_of_kart))
            action['acceleration'] = .3
            action['brake'] = True
            action['drift'] = True

        if abs(x) < .2 and abs(y) < .2:
            #print(Fore.MAGENTA + 'hammer time!' + Fore.WHITE)
            if facing_angle < attack_cone / 2: # within attack cone - hammer it home!
                action['steer'] = 0
                action['acceleration'] = 1
            # else: # steer towards attack cone
            #     action['steer'] = np.sign(vector_center[0])
            #     action['acceleration'] = .5


        # if np.sign(x) != np.sign(action['steer']):
            #print(Fore.RED + 'error: turned wrong way - x: {}, steer: {}'.format(x, action['steer']) + Fore.WHITE)

        return action

    # ============================================= defense logic =============================================
    # def defense(self):
    #     if np.sign(self.our_goal_center[1]) == np.sign(self.kart_loc[1]):
    #         #print("WE PLAYIN DEFENSE")
    #         #print("WE PLAYIN DEFENSE")
    #         return True
    #     else:
    #         return False

    # def defense_action(self, puck_loc, action):
    #     kart_front_x, kart_front_y = self.kart_front
    #     kart_x, kart_y = self.kart_loc
    #     face_right = np.sign(self.kart_loc[1]) * (kart_front_x - kart_x)
    #     # face_left =
    #     # face_up =
    #     # face_down =
    #     goal_x = abs(self.our_goal_left[0])
    #     in_front_of_you = self.puck_loc[0] < .1 and self.puck_loc[0] > -.1 and self.puck_loc[1] < .2 and self.puck_loc[1] > -.2
    #     to_your_right = self.puck_loc[0] > .1
    #     to_your_left = self.puck_loc[0] < -.1
        # If you're on the right of the puck and not lined up, line up
        # if (not in_front_of_you) and (kart_x:
        # If you're on the left of the puck and not lined up, line up

        # If you're "below" the puck and not lined, line up

        #If you're "on top" of the puck, frikin good luck and try to line up.

        # If you're lined up, go straight, go straight

        # x = puck_loc[0]
        # if x > .3 or x < -.3:
        #     action['steer'] = np.sign(x) * 1
        # else:
        #     action['steer'] = 0
        # if x > .4 or x < -.4:
        #     action['steer'] = np.sign(x) * 1
        #     action['drift'] = True
        #     action['acceleration'] = .5
        # else:
        #     action['drift'] = False
        #     return action

    # ============================================= helper functions =============================================
    # Convert a location to a numpy array
    @staticmethod
    def to_numpy(location):
        """
        Don't care about location[1], which is the height
        """
        return np.float32([location[0], location[2]])

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

    def check_reset(self, kart_loc):
        threshold = 5
        last_loc = self.past_kart_locs[-1]
        x_diff = abs(last_loc[0] - kart_loc[0])
        y_diff = abs(last_loc[-1] - kart_loc[-1])
        if x_diff > threshold or y_diff > threshold:
            # #print('reset check true', x_diff, y_diff)
            return True
        # #print('reset check false', x_diff, y_diff)
        return False
