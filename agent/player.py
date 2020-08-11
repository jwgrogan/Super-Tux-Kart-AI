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
        # load model
        self.model = load_model()
        self.model.eval()
        # Select Team
        self.team = player_id % 2
        # Determine whether player is on offense (position = 0) or defense (position = 1)
        if self.team == 0:
            self.position = (player_id / 2) % 2
        else:
            self.position = (player_id - 1 / 2) % 2

    def act(self, image, player_info):
        action = {'acceleration': 1, 'brake': False, 'drift': False, 'nitro': False, 'rescue': False, 'steer': 0}
        # Puck Information
        image_transform = F.to_tensor(image)[None]
        puck_location = (self.model(image_transform).detach().cpu().numpy())[0]
        # print("Puck Location", puck_location)
        x = puck_location[0]
        y = puck_location[1]

        x_middle = 0
        # Kart Information
        kart = to_numpy(player_info.kart.location)
        kart_front = to_numpy(player_info.kart.front)
        kart_velocity = player_info.kart.velocity
        # Try Out Turning
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
        return action, puck_location
