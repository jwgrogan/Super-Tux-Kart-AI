import numpy as np
from .model import PuckDetector
from .utils import load_model
import torch

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
       all_players = ['adiumy', 'amanda', 'beastie', 'emule', 'gavroche', 'gnu', 'hexley', 'kiki', 'konqi', 'nolok', 'pidgin', 
                    'puffy', 'sara_the_racer', 'sara_the_wizard', 'suzanne', 'tux', 'wilber', 'xue']
    """
    kart = ''

    
    def __init__(self, player_id = 0):
        """
        Set up a soccer player.
        The player_id starts at 0 and increases by one for each player added. You can use the player id to figure out your team 
        (player_id % 2), or assign different roles to different agents.
        """
        from os import path

        # select kart
        self.kart = 'xue'
        # load model
        self.model = load_model().eval()

        # Select Team
        self.team = player_id % 2
        # Determine whether player is on offense (position = 0) or defense (position = 1)
        if (self.team == 0):
            self.position = (player_id / 2) % 2
        else:
            self.position = (player_id - 1 / 2 ) % 2

        self.teammate_has_puck = False
        self.step = 0
        
        

    def act(self, image, player_info):
        """
        Set the action given the current image
        :param image: numpy array of shape (300, 400, 3)
        :param player_info: pystk.Player object for the current kart -> https://pystk.readthedocs.io/en/latest/state.html?highlight=pystk.Player#pystk.Player
        return: Dict describing the action
        """
        action = {'acceleration': 0, 'brake': False, 'drift': False, 'nitro': False, 'rescue': False, 'steer': 0}
        acc, brake, steer, drift, nitro = 0, False, 0, False, False
        """
        Your code here.
        """
        target_vel = 10.0
        image = torch.from_numpy(image)
        ball = self.model(image.reshape([image.shape[2], image.shape[0], image.shape[1]]))
        # print("BALL", ball[0, 0])
        # get kart location
        kart_front = player_info.kart.front
        current_vel = np.linalg.norm(player_info.kart.velocity)
        kart_x = kart_front[0]
        kart_y = kart_front[2]
        ball_x = ball[0,0]
        ball_y = ball[0,1]

        # if ball == None:
        #

        # import csv
        # with open('save/ball_locations.csv', 'r') as f:
        #     for row in reversed(list(csv.reader(f))):
        #         ball_location = row 
        #         break

        ratio = current_vel / target_vel
        if ratio <= 0.5:
            acc_mult = 1
        elif ratio <= 0.7:
            acc_mult = 0.8
        elif ratio <= 0.95:
            acc_mult = 0.7
        else:
            # print("are we here?")
            acc_mult = 0
        # print("AIM POINT", aim_point, current_vel)

        # Turn on drift for wide turns. i.e. when aimpoint is >0.5 on the ball_y axis
        # Acceleration to 1 if point is between -0.1 and 0.1
        if ball_x > -0.1 and ball_x < 0.1:
            acc = 1.0 * acc_mult
            # nitro = True
        # 0.8 if between -0.1 and -0.3 or 0.1 and 0.3
        elif (ball_x <= -0.1 and ball_x > -0.25) or (ball_x >= 0.1 and ball_x < 0.25):
            acc = 0.8 * acc_mult
            if ball_x < 0:
                steer = -0.8
            else:
                steer = 0.8
            # nitro = True
        else:
            # acc = 0.2 * acc_mult
            drift = True
            if ball_x > 0:
                steer = 1.0
            else:
                steer = -1.0

        if current_vel >= (1.2 * target_vel):
            acc = 0
            # nitro = False
            brake = True

        action['acceleration'] = acc
        action['brake'] = brake
        action['steer'] = steer
        action['drift'] = drift
        action['nitro'] = nitro
        
        if (self.team == 0):
            print("step", self.step)
            self.step += 1
            print("Ball cord: {}  Kart Cord: {}".format(ball, kart_front))
            print("velocity", current_vel)
            print("action", action)

        return action


