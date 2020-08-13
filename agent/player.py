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
    print("sup motherfuckers")
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
    self.past_locs = deque(maxlen=10)
    self.past_state = deque(maxlen=10)
    self.past_actions = deque(maxlen=10)
    self.state_lock = False
    self.state_lock_turns = 0
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
    action = {'acceleration': 1, 'brake': False, 'drift': False, 'nitro': False, 'rescue': False, 'steer': 0}
    # Puck Information
    image_transform = F.to_tensor(image)[None]
    # puck_loc = self.model(image_transform).squeeze(0).detach().cpu().numpy()
    image_puck_loc = (self.model(image_transform).detach().cpu().numpy())[0]
    # print("Puck Location", puck_location)
    x = image_puck_loc[0]
    y = image_puck_loc[1]

    puck_loc = self.image_to_local(x, y, player_info)
    # print('puck loc:', puck_loc)


    x_middle = 0
    # Kart Information
    self.kart_loc = self.to_numpy(player_info.kart.location)

    self.kart_front = self.to_numpy(player_info.kart.front)
    print('kart loc', self.kart_loc, self.kart_front)
    print('ANGLEEEE', np.arctan2([self.kart_loc[1], self.kart_front[1]],[self.kart_loc[0], self.kart_front[0]]) * 180/np.pi)
    kart_velocity = player_info.kart.velocity
    kart_attachment = player_info.kart.attachment.type
    kart_powerup = player_info.kart.powerup.type
    # goal location
    self.set_goal_loc(self.kart_loc)

    # action['acceleration'] = .5
    # action['steer'] = np.sign(x) * 1

    # set kart state
    if not self.state_lock:
      self.state = self.set_state(self.kart_loc, puck_loc)
      # self.stuck(kart_loc)
      # self.state = 'stuck'
      print(self.state)
      if self.state == 'kickoff':
        puck_loc = (0, 0)
        action = self.kickoff_action(self.kart_loc, self.kart_front, puck_loc, action)
      elif self.state == 'in_goal':
        action = self.getOutOfGoal(self.kart_loc, action, player_info)
      elif self.state == 'attack':
        action = self.attack_action(self.kart_loc, self.kart_front, puck_loc, x, action)
      elif self.state == 'positioning':
        action = self.positioning_action(self.kart_loc, self.kart_front, puck_loc, action)
      elif self.state == 'stuck':
        action = self.stuck_action(self.kart_loc, action)

    else:
      self.state_lock_turns -= 1
      action = self.past_actions[-1]

    if self.state_lock_turns == 0:
      self.state_lock = False

    self.past_locs.append(self.kart_loc)
    self.past_state.append(self.state)
    self.past_actions.append(action)
    # self.position = (player_id - 1 / 2 ) % 2
    # self.teammate_has_puck = False
    # self.step = 0

    # TODO: Remove puck_loc from the returned before submitting
    return action, image_puck_loc

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

  # Gets vector between two objects given both their in game locations.
  @staticmethod
  def get_vector_from_this_to_that(me, obj, normalize=True):
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
    return (x, y)

  # Used to Set the goal locations for each player in the beggining
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
    # set kickoff and start timer for end
    if len(self.past_locs) == 0 or self.kickoff(kart_loc) == True:
      self.kickoff_timer = 0
      self.set_goal_loc(kart_loc)
      return 'kickoff'
    else:
      self.kickoff_timer += 1

    if self.state == 'kickoff' and self.kickoff_timer < 30:
      return 'kickoff'
    if self.stuck(kart_loc) == True:
      return 'stuck'
    if self.inGoal(kart_loc) == True:
      return 'in_goal'
    elif self.positioning(kart_loc, puck_loc) == True:
      return 'positioning'
    elif self.attack(kart_loc, puck_loc) == True:
      return 'attack'

  def inGoal(self, kart_loc):
    if ((kart_loc[1] > 66) or (kart_loc[1] < -66)):
      self.state_lock = True
      self.state_lock_turns = 10
      return True
    else:
      return False

  def getOutOfGoal(self, kart_loc, action, playerinfo):
    past_action = self.past_actions[0]
    distance_moved = abs(kart_loc[1]) - abs(self.past_locs[0][1])
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
      if (self.kart_front[1] - self.kart_loc[1] > .3):
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

    # print('distance moved', distance_moved)
    # print('perspective?', self.image_to_local(0,0,playerinfo))
    # Check to make sure kart is facing the outside of the goal.
    if distance_moved < 0:
      return past_action
    # Otherwise change up how you move. Ideally want to make a K turn out
    else:
      action['brake'] = not past_action['brake']

      if past_action['acceleration'] > 0:
        action['acceleration'] = 0
      else:
        action['acceleration'] = 1

      if (past_action['steer'] > 0):
        action['steer'] = -1
      else:
        action['steer'] = 1

      return action

  def stuck(self, kart_loc):
    if ((abs(kart_loc - self.past_locs[-1]) < 0.05).all()):
      self.state_lock = True
      self.state_lock_turns = 3
      return True
    else:
      return False


  def stuck_action(self, kart_loc, action):
    past_action = self.past_actions[4]
    distance_moved = kart_loc - self.past_locs[0]

    action['brake'] = not past_action['brake']

    if past_action['acceleration'] > 0:
      action['acceleration'] = 0
    else:
      action['acceleration'] = .5

    if(np.sign(distance_moved[0]) == np.sign(distance_moved[1])):
      action['steer'] = 1
    else:
      action['steer'] = -1


    print("IMAAAAGINE", action['steer'])
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
    action = {'acceleration': 1, 'steer': 0, 'brake': False, 'nitro': True}

    distance_to_puck = np.linalg.norm(kart_loc - puck_loc)
    # # print(distance_to_puck)
    # vector_of_kart = self.get_vector_from_this_to_that(kart_loc, kart_front)
    # vector_to_goal = self.get_vector_from_this_to_that(kart_loc, self.their_goal)
    # vector_to_puck = self.get_vector_from_this_to_that(kart_loc, puck_loc)

    # # action['steer'] = np.sign(np.dot(vector_of_kart, vector_to_puck))
    # action['acceleration'] = 1
    # action['nitro'] = True

    if distance_to_puck < 20:
      action['steer'] = 0
      action['acceleration'] = .5
      action['nitro'] = False

    return action

  def positioning(self, kart_loc, puck_loc):
    distance_to_puck = np.linalg.norm(kart_loc - puck_loc)
    kart_to_their_goal = np.linalg.norm(kart_loc - np.float32(self.their_goal))
    print(kart_to_their_goal)
    puck_to_their_goal = np.linalg.norm(puck_loc - np.float32(self.their_goal))
    print(puck_to_their_goal)

    if abs(kart_to_their_goal) > abs(puck_to_their_goal):
      print('positioning: True')
      return True
    else:
      print('positioning: False')
      return False

  def positioning_action(self, kart_loc, kart_front, puck_loc, action):
    vector_of_kart = self.get_vector_from_this_to_that(kart_loc, kart_front)
    vector_to_goal = self.get_vector_from_this_to_that(kart_loc, self.their_goal)
    vector_to_puck = self.get_vector_from_this_to_that(kart_loc, puck_loc)

    action['brake'] = True
    action['acceleration'] = 0
    # action['steer'] = -np.sign(np.dot(vector_of_kart, vector_to_puck))
    return action

  def attack(self, kart_loc, puck_loc):
    distance_to_puck = np.linalg.norm(kart_loc - puck_loc)
    kart_to_their_goal = np.linalg.norm(kart_loc - np.float32(self.their_goal))
    puck_to_their_goal = np.linalg.norm(puck_loc - np.float32(self.their_goal))

    if abs(kart_to_their_goal) > abs(puck_to_their_goal):
      print('attack: True')
      return True
    else:
      print('attack: False')
      return False

  def attack_action(self, kart_loc, kart_front, puck_loc, action):
    action = {'acceleration': 1, 'steer': 0, 'brake': False}

    puck_loc = np.float32(puck_loc)
    distance_to_puck_ = np.linalg.norm(kart_loc - puck_loc)

    vector_of_kart = self.get_vector_from_this_to_that(kart_loc, kart_front)
    vector_to_goal = self.get_vector_from_this_to_that(kart_loc, self.their_goal)
    vector_to_puck = self.get_vector_from_this_to_that(kart_loc, puck_loc)

    action['steer'] = np.sign(np.dot(vector_of_kart, vector_to_puck))
    x = puck_loc[0]
    if x > .3 or x < -.3:
      action['steer'] = np.sign(np.dot(vector_of_kart, vector_to_puck))
    else:
      action['steer'] = np.sign(np.dot(vector_of_kart, vector_to_puck))
    if x > .4 or x < -.4:
      action['steer'] = np.sign(np.dot(vector_of_kart, vector_to_puck))
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
