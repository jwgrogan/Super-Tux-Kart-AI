import numpy as np
from .model import PuckDetector, load_model
import torchvision.transforms.functional as F
import torch


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

class HockeyPlayer:
    def __init__(self, player_id=0):
        # select kart
        self.kart = 'xue'
        # set up parameters
        self.state = 'start'
        # self.states = {'kickoff': self.kickoff_action,
        #                 'search': self.search_action,
        #                 'attack': self.attack_action,
        #                 'stuck': self.stuck_action
        #                 }
        from collections import deque
        self.past_locs = deque(maxlen=10)
        self.past_actions = deque(maxlen=10)
                    
        self.team_goal = (0, -64)
        self.opponent_goal = (0, 64)
        # load model
        self.model = load_model().eval()

        # Select Team
        self.team = player_id % 2
        # Determine whether player is on offense (position = 0) or defense (position = 1)
        if self.team == 0:
            self.position = (player_id / 2) % 2
        else:
            self.position = (player_id - 1 / 2) % 2

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

    def set_state(self, kart_loc):
        """
        set current state of the kart
        """
        if self.past_locs == None or self.kickoff(kart_loc) == True:
            self.state = 'kickoff'
        elif self.stuck(kart_loc):
            self.state == 'stuck'
        elif self.search(kart_loc) == True:
            self.state = 'search'
        else:
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
        for past_locs in reversed(self.past_locs):
            x_diff = abs(past_locs[0] - kart_loc[0])
            z_diff = abs(past_locs[-1] - kart_loc[-1])
            if x_diff > threshold or z_diff > threshold:
                return True
        return False

    def kickoff_action(self, action):
        action['acceleration'] = 1
        action['nitro'] = True
        return action

    def search(self, kart_loc):
        if self.kickoff(kart_loc) == False:
            return True
        return False

    def search_action(self, x, y, action):
        if x == 0 or x == 2 or y == 0 or y == 2:
            action['brake'] = True
            action['acceleration'] = 0
        else:
            if x > .3 or x < -.3:
                action['steer'] = np.sign(x) * 1
                print("STEEEEERR", action['steer'])
            else:
                action['steer'] = 0
            if x > .5 or x < -.5:
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
        puck_location = (self.model(image_transform).detach().cpu().numpy())[0]
        # print("Puck Location", puck_location)
        x = puck_location[0]
        y = puck_location[1]

        x_middle = 0
        # Kart Information
        kart_loc = to_numpy(player_info.kart.location)
        kart_front = to_numpy(player_info.kart.front)
        kart_velocity = player_info.kart.velocity
        kart_attachment = player_info.kart.attachment.type
        kart_powerup = player_info.kart.powerup.type
        
        # set kart state
        self.state = self.set_state(kart_loc)
        if self.state == 'kickoff':
            action = self.kickoff_action(action)
        elif self.state == 'stuck':
            action = self.stuck_action(x, action)
        elif self.state == 'search':
            action = self.search_action(x, y, action)
        # elif self.state == 'attack':
        #     action = self.attack_action(action)
        
        self.past_locs.append(kart_loc)
        self.past_actions.append(action)
        # self.position = (player_id - 1 / 2 ) % 2
        # self.teammate_has_puck = False
        # self.step = 0

        return action, puck_location
        
        
        
    
    




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
