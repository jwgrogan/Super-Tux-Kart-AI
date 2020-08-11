import pystk
import numpy as np
import csv


class Player:
  def __init__(self, player, team=0):
    self.player = player
    self.team = team

  @property
  def config(self):
    return pystk.PlayerConfig(controller=pystk.PlayerConfig.Controller.AI_CONTROL, kart=self.player.kart,
                              team=self.team)

  def __call__(self, image, player_info):
    return self.player.act(image, player_info)



class Tournament:
  _singleton = None

  def __init__(self, players, screen_width=400, screen_height=300, track='icy_soccer_field'):
    assert Tournament._singleton is None, "Cannot create more than one Tournament object"
    Tournament._singleton = self

    self.graphics_config = pystk.GraphicsConfig.hd()
    self.graphics_config.screen_width = screen_width
    self.graphics_config.screen_height = screen_height
    pystk.init(self.graphics_config)

    self.race_config = pystk.RaceConfig(num_kart=len(players), track=track, mode=pystk.RaceConfig.RaceMode.SOCCER,
                                        difficulty=2)
    self.race_config.players.pop()

    self.active_players = []
    for p in players:
      if p is not None:
        self.race_config.players.append(p.config)
        self.active_players.append(p)

    self.k = pystk.Race(self.race_config)

    self.k.start()
    self.k.step()

  def get_vector_from_this_to_that(self, me, obj, normalize=True):
    """
    Expects numpy arrays as input
    """

    vector = obj - me

    if normalize:
      return vector / np.linalg.norm(vector)

    return vector

  def to_numpy(self, location):
      """
      Don't care about location[1], which is the height
      """
      return np.float32([location[0], location[1], location[2]])

  @staticmethod
  def _to_image(x, proj, view):
    p = proj @ view @ np.array(list(x) + [1])
    return np.clip(np.array([p[0] / p[-1], -p[1] / p[-1]]), -1, 1)

  def play(self, save=None, max_frames=50, verbose=False):

    if verbose:
      import matplotlib.pyplot as plt
      fig, ax = plt.subplots(1, 1)

    state = pystk.WorldState()
    if save is not None:
      import PIL.Image
      import os
      if not os.path.exists(save):
        os.makedirs(save)
    for t in range(max_frames):
      print('\rframe %d' % t, end='\r')

      state.update()

      list_actions = []
      for i, p in enumerate(self.active_players):
        player = state.players[i]
        image = np.array(self.k.render_data[i].image)

        action = pystk.Action()
        player_action = p(image, player)
        for a in player_action:
          setattr(action, a, player_action[a])

        list_actions.append(action)

        if save is not None:
          # print("porque")
          PIL.Image.fromarray(image).save(os.path.join(save, 'player%02d_%05d.png' % (i, t)))
          ball_coords = self.to_numpy(state.soccer.ball.location)
          kart_proj = np.array(state.players[i].camera.projection).T
          kart_view = np.array(state.players[i].camera.view).T
          kart_ball_coords = self._to_image(ball_coords, kart_proj, kart_view)
          # np.savez(os.path.join(save, ('ball%02d' % i)), kart_ball_view)
          with open(save + '/player%02d_%05d.csv'% (i, t), mode='w') as ball_file:
            ball_writer = csv.writer(ball_file, delimiter=',')
            ball_writer.writerow(kart_ball_coords)

        if verbose:
          ax.clear()
          ax.imshow(self.k.render_data[0].image)
          WH2 = np.array([self.graphics_config.screen_width, self.graphics_config.screen_height]) / 2
          # blue circle on player kart
          ax.add_artist(
            plt.Circle(WH2 * (1 + self._to_image(state.players[i].kart.location, kart_proj, kart_view)), 2, ec='b', fill=False, lw=1.5))
          # red circle on puck
          ax.add_artist(
            plt.Circle(WH2 * (1 + self._to_image(ball_coords, kart_proj, kart_view)), 2, ec='r', fill=False, lw=1.5))
          # green circle on items
          for j, item in enumerate(state.items):
            ax.add_artist(
              plt.Circle(WH2 * (1 + self._to_image(item.location, kart_proj, kart_view)), 2, ec='g', fill=False, lw=1.5))
          # yellow circle on other players
          for j, player in enumerate(self.active_players):
            if j != i:
              ax.add_artist(
                plt.Circle(WH2 * (1 + self._to_image(state.players[j].kart.location, kart_proj, kart_view)), 2, ec='y', fill=False, lw=1.5))
          plt.pause(1e-3)
          

      s = self.k.step(list_actions)
      if not s:  # Game over
        break

    if save is not None:
      import subprocess
      for i, p in enumerate(self.active_players):
        dest = os.path.join(save, 'player%02d' % i)
        output = save + '_player%02d.mp4' % i
        subprocess.call(['ffmpeg', '-y', '-framerate', '10', '-i', dest + '_%05d.png', output], shell=True)
    if hasattr(state, 'soccer'):
      return state.soccer.score
    return state.soccer_score

  def close(self):
    self.k.stop()
    del self.k
