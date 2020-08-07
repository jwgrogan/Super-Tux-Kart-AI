import numpy as np
from .model import Model
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
        # select kart
        self.kart = 'xue'
        # load model
        from os import path
        self.model = torch.load(path.join(path.dirname(path.abspath(__file__)), 'model.th'))
        # select behavior
        self.team = player_id % 2
        if player_id < 2:
            self.defense = True
        else:
            self.offense = True
        
        self.teammate_has_puck = False
        
        

    def act(self, image, player_info):
        """
        Set the action given the current image
        :param image: numpy array of shape (300, 400, 3)
        :param player_info: pystk.Player object for the current kart -> https://pystk.readthedocs.io/en/latest/state.html?highlight=pystk.Player#pystk.Player
        return: Dict describing the action
        """
        action = {'acceleration': 0, 'brake': False, 'drift': False, 'nitro': False, 'rescue': False, 'steer': 0}
        """
        Your code here.
        """
        image = torch.from_numpy(image)
        ball = self.model.forward(self, image)
        # get kart location
        kart_front = player_info.kart.front
        kart_x = kart_front[0]
        kart_z = kart_front[2]

        if ball == None:
            

        # import csv
        # with open('save/ball_locations.csv', 'r') as f:
        #     for row in reversed(list(csv.reader(f))):
        #         ball_location = row 
        #         break

        ball_x = ball[0]
        ball_z = ball[2]

        # if current_vel != target_vel:
        #     action.acceleration = 1

        # if x == 0:
        #     action.nitro = True
        
        if x < 0: #aim_point is to the left
            if x < -0.5:
                action['drift'] = True
                action['steer'] = -1
                action['acceleration'] = .25
                action['brake'] = True
            elif x < -0.3:
                action['drift'] = True
                action['steer'] = -1
                action['acceleration'] = .75
            else:
                action['steer'] = -.7

        elif x > 0: #aim_point is to the right
            if x > 0.5:
                action['drift'] = True
                action['steer'] = 1
                action['acceleration'] = .25
                action['brake'] = True
            elif x > 0.3:
                action['drift'] = True
                action['steer'] = 1
                action['acceleration'] = .75
            else:
                action['steer'] = .7


        return action


