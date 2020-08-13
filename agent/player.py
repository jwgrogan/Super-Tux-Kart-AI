import numpy as np
from .model import PuckDetector, load_model
import torchvision.transforms.functional as F
import torch

# helper functions
def to_numpy(location):
    """
    Don't care about location[1], which is the height
    """
    return np.float32([location[0], location[2]])

def _to_image(x, proj, view):
    W, H = 400, 300
    p = proj @ view @ np.array(list(x) + [1])
    return np.array([W / 2 * (p[0] / p[-1] + 1), H / 2 * (1 - p[1] / p[-1])])

def get_vector_from_this_to_that(me, obj, normalize=True):
    """
    Expects numpy arrays as input
    """
    vector = obj - me
    if normalize:
        return vector / np.linalg.norm(vector)
    return vector

def _to_world(x, y, proj, view, height=0):
        W, H = 400, 300
        pv_inv = np.linalg.pinv(proj @ view)
        xy, d = pv_inv.dot([float(x) / (W / 2) - 1, 1 -
                            float(y) / (H / 2), 0, 1]), pv_inv[:, 2]
        x0, x1 = xy[:-1] / xy[-1], (xy+d)[:-1] / (xy+d)[-1]
        t = (height-x0[1]) / (x1[1] - x0[1])
        if t < 1e-3 or t > 10:
            # Project the point forward by a certain distance, if it would end up behind
            t = 10
        return t * x1 + (1-t) * x0

def image_to_local(x, y, player):
        y += 40
        proj = np.array(player.camera.projection).T
        view = np.array(player.camera.view).T
        x, _, y = self._to_world(x, y, proj, view, player.kart.location[1])
        # strip ridiculous values
        if abs(x) > 39 or abs(y) > 64:
            return (99, 99)
        return (x, y)

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
        # select kart
        self.kart = 'xue'
        # set up parameters
        self.state = 'kickoff'
        # self.states = {'kickoff': self.kickoff_action,
        #                 'positioning': self.positioning_action,
        #                 'attack': self.attack_action,
        #                 'stuck': self.stuck_action
        #                 }
        from collections import deque
        self.past_locs = deque(maxlen=10)
        self.past_actions = deque(maxlen=10)
        
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

    def to_numpy(self, location):
        """
        Don't care about location[1], which is the height
        """
        return np.float32([location[0], location[2]])

    def _to_image(self, x, proj, view):
        W, H = 400, 300
        p = proj @ view @ np.array(list(x) + [1])
        return np.array([W / 2 * (p[0] / p[-1] + 1), H / 2 * (1 - p[1] / p[-1])])

    def get_vector_from_this_to_that(self, me, obj, normalize=True):
        """
        Expects numpy arrays as input
        """
        vector = obj - me
        if normalize:
            return vector / np.linalg.norm(vector)
        return vector
    
    
    def _to_world(self, x, y, proj, view, height=0):
        W, H = 400, 300
        pv_inv = np.linalg.pinv(proj @ view)
        xy, d = pv_inv.dot([float(x) / (W / 2) - 1, 1 -
                            float(y) / (H / 2), 0, 1]), pv_inv[:, 2]
        x0, x1 = xy[:-1] / xy[-1], (xy+d)[:-1] / (xy+d)[-1]
        t = (height-x0[1]) / (x1[1] - x0[1])
        if t < 1e-3 or t > 10:
            # Project the point forward by a certain distance, if it would end up behind
            t = 10
        return t * x1 + (1-t) * x0

    def image_to_local(self, x, y, player):
        y += 40
        proj = np.array(player.camera.projection).T
        view = np.array(player.camera.view).T
        x, _, y = self._to_world(x, y, proj, view, player.kart.location[1])
        # strip ridiculous values
        if abs(x) > 39 or abs(y) > 64:
            return (99, 99)
        return (x, y)

    def set_goal_loc(self, kart_loc):
        z = kart_loc[-1]
        if z < 0:
            self.our_goal = (0, -64)
            self.their_goal = (0, 64)
        else:
            self.our_goal = (0, 64)
            self.their_goal = (0, -64)

    def set_state(self, kart_loc, puck_loc):
        """
        set current state of the kart
        """
        if len(self.past_locs) == 0 or self.kickoff(kart_loc) == True:
            self.state = 'kickoff'
            self.kickoff_timer = 0
        else:
            self.kickoff_timer += 1

        if self.state == 'kickoff' and self.kickoff_timer < 60:
            self.state = 'kickoff'

        if self.stuck(kart_loc) == True:
            self.state == 'stuck'
        elif self.positioning(kart_loc) == True:
            self.state = 'positioning'
        elif self.attack_position(kart_loc, puck_loc) == True:
            self.state = 'attack'
        return self.state
    
    def stuck(self, kart_loc):
        threshold = 1
        for past_locs in reversed(self.past_locs):
            x_diff = abs(past_locs[0] - kart_loc[0])
            z_diff = abs(past_locs[-1] - kart_loc[-1])
            if x_diff > threshold or z_diff > threshold:
                return False
        return True

    def stuck_action(self, x, action):
        action['brake'] = True
        action['acceleration'] = 0
        action['steer'] = np.sign(x) * 1
        return action
        
    def kickoff(self, kart_loc):
        threshold = 5
        last_loc = self.past_locs[-1]
        x_diff = abs(last_loc[0] - kart_loc[0])
        z_diff = abs(last_loc[-1] - kart_loc[-1])
        if x_diff > threshold or z_diff > threshold:
            return True
        return False

    def kickoff_action(self, kart_loc, kart_front, puck_loc, action):
        """
        boost towards center to find puck
        """
        vector_of_kart = get_vector_from_this_to_that(kart_loc, kart_front)
        vector_to_goal = get_vector_from_this_to_that(kart_loc, self.their_goal)
        vector_to_puck = get_vector_from_this_to_that(kart_loc, puck_loc)

        action['steer'] = np.sign(np.dot(vector_of_kart, vector_to_puck))
        action['acceleration'] = 1
        action['nitro'] = True
        
        return action

    def positioning(self, kart_loc):
        x = kart_loc[0]
        y = kart_loc[-1]
        if x == 0 or x == 2 or y == 0 or y == 2:
            return True
        return False

    def positioning_action(self, action):
        action['brake'] = True
        action['acceleration'] = 0
        return action   

    def attack_position(self, kart_loc, puck_loc):
        kart_to_their_goal = np.linalg.norm(kart_loc - np.float32(self.their_goal))
        puck_to_their_goal = np.linalg.norm(puck_loc - np.float32(self.their_goal))

        if abs(kart_to_their_goal) > abs(puck_to_their_goal):
            return True
        else:
            return False

    def attack_action(self, kart_loc, kart_front, puck_loc, x, action):
        
        kart_to_puck = np.linalg.norm(kart_loc - puck_loc)

        vector_of_kart = get_vector_from_this_to_that(kart_loc, kart_front)
        vector_to_goal = get_vector_from_this_to_that(kart_loc, self.their_goal)
        vector_to_puck = get_vector_from_this_to_that(kart_loc, puck_loc)

        action['steer'] = np.sign(0) * 1#aim towards center
        # x = puck_loc[0]
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


    def act(self, image, player_info):
        """
        Set the action given the current image
        :param image: numpy array of shape (300, 400, 3)
        :param player_info: pystk.Player object for the current kart -> https://pystk.readthedocs.io/en/latest/state.html?highlight=pystk.Player#pystk.Player
        return: Dict describing the action
        """
        action = {'acceleration': 1, 'brake': False, 'drift': False, 'nitro': False, 'rescue': False, 'steer': 0}
        # Puck Information
        image_transform = F.to_tensor(image)[None]
        image_puck_loc = (self.model(image_transform).detach().cpu().numpy())[0]
        
        # print("Puck Location", puck_location)
        x = image_puck_loc[0]
        y = image_puck_loc[1]
        
        puck_loc = np.float32(self.image_to_local(x, y, player_info))
        print(puck_loc)

        x_middle = 0
        # Kart Information
        kart_loc = to_numpy(player_info.kart.location)
        print(kart_loc.shape())     

        kart_front = to_numpy(player_info.kart.front)
        kart_velocity = player_info.kart.velocity
        kart_attachment = player_info.kart.attachment.type
        kart_powerup = player_info.kart.powerup.type
        # goal location
        self.set_goal_loc(kart_loc)

        # action['acceleration'] = .5
        # action['steer'] = np.sign(x) * 1

        # set kart state
        self.state = self.set_state(kart_loc, puck_loc)
        print(self.state)

        if self.state == 'kickoff':
            puck_loc = (0, 0, 0)
            action = self.kickoff_action(kart_loc, kart_front, puck_loc, action)
        elif self.state == 'attack':
            action = self.attack_action(kart_loc, kart_front, puck_loc, x, action)
        elif self.state == 'positioning':
            action = self.positioning_action(action)
        elif self.state == 'stuck':
            action = self.stuck_action(x, action)
        
        self.past_locs.append(kart_loc)
        self.past_actions.append(action)

        # self.position = (player_id - 1 / 2 ) % 2
        # self.teammate_has_puck = False
        # self.step = 0

        return action, image_puck_loc
        
        
        
    
    




    # def act(self, image, player_info):
    #     """
    #     Set the action given the current image
    #     :param image: numpy array of shape (300, 400, 3)
    #     :param player_info: pystk.Player object for the current kart -> https://pystk.readthedocs.io/en/latest/state.html?highlight=pystk.Player#pystk.Player
    #     return: Dict describing the action
    #     """
    #     action = {'acceleration': 0, 'brake': False, 'drift': False, 'nitro': False, 'rescue': False, 'steer': 0}
    #     acc, brake, steer, drift, nitro = 0, False, 0, False, False
    #     """
    #     Your code here.
    #     """
    #     # # grab comms
    #     # import . from comms
    #     # team_loc = comms.team_loc

    #     # 

    #     # determine ball location
    #     image = torch.from_numpy(image)
    #     ball = self.model(image.reshape([image.shape[2], image.shape[0], image.shape[1]]))
    #     ball_x = ball[0,0]
    #     ball_y = ball[0,1]
    #     # print("BALL", ball[0, 0])
    #     # get kart information
    #     kart_loc = player_info.kart.location
    #     current_vel = np.linalg.norm(player_info.kart.velocity)
    #     attachment = player_info.kart.attachment.type
    #     powerup = player_info.kart.powerup.type

    #     if self.state = 'kickoff':

    #     elif self.state == 'stuck'

    #     elif self.state = 'search':
    #         action = search_action(ball_x, ball_y, current_vel, target_vel=10.0)
        
    #     elif self.state = 'attack'
        
        
    #     if (self.team == 0):
    #         print("step", self.step)
    #         self.step += 1
    #         print("Ball cord: {}  Kart Cord: {}".format(ball, kart_loc))
    #         print("velocity", current_vel)
    #         print("action", action)
        
    #     self.past_locs.append(kart_loc)
    #     self.past_actions.append(action)

    #     return action




# else:
#             ratio = current_vel / target_vel
#             if ratio <= 0.5:
#                 acc_mult = 1
#             elif ratio <= 0.7:
#                 acc_mult = 0.8
#             elif ratio <= 0.95:
#                 acc_mult = 0.7
#             else:
#                 # print("are we here?")
#                 acc_mult = 0
#             # print("AIM POINT", aim_point, current_vel)

#             # Turn on drift for wide turns. i.e. when aimpoint is >0.5 on the ball_y axis
#             # Acceleration to 1 if point is between -0.1 and 0.1
#             if ball_x > -0.1 and ball_x < 0.1:
#                 acc = 1.0 * acc_mult
#                 # nitro = True

#             # 0.8 if between -0.1 and -0.3 or 0.1 and 0.3
#             elif (ball_x <= -0.1 and ball_x > -0.25) or (ball_x >= 0.1 and ball_x < 0.25):
#                 acc = 0.8 * acc_mult
#                 if ball_x < 0:
#                     steer = -0.8
#                 else:
#                     steer = 0.8
#                 # nitro = True

#             else:
#                 # acc = 0.2 * acc_mult
#                 drift = True
#                 if ball_x > 0:
#                     steer = 1.0
#                 else:
#                     steer = -1.0

#             if current_vel >= (1.2 * target_vel):
#                 acc = 0
#                 # nitro = False
#                 brake = True

#         action['acceleration'] = acc
#         action['brake'] = brake
#         action['steer'] = steer
#         action['drift'] = drift
#         action['nitro'] = nitro
