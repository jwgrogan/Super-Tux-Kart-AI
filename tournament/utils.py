import pystk
import numpy as np
# TODO: remove this before submitting
import csv

class Player:
    def __init__(self, player, team=0):
        self.player = player
        self.team = team

    @property
    def config(self):
        return pystk.PlayerConfig(controller=pystk.PlayerConfig.Controller.PLAYER_CONTROL, kart=self.player.kart, team=self.team)
    
    def __call__(self, image, player_info):
        return self.player.act(image, player_info)


class Tournament:
    _singleton = None

    #TODO: Remove before submitting
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
    #TODO: END

    def __init__(self, players, screen_width=400, screen_height=300, track='icy_soccer_field'):
        assert Tournament._singleton is None, "Cannot create more than one Tournament object"
        Tournament._singleton = self

        self.graphics_config = pystk.GraphicsConfig.hd()
        self.graphics_config.screen_width = screen_width
        self.graphics_config.screen_height = screen_height
        pystk.init(self.graphics_config)

        self.race_config = pystk.RaceConfig(num_kart=len(players), track=track, mode=pystk.RaceConfig.RaceMode.SOCCER, difficulty=2)
        self.race_config.players.pop()
        
        self.active_players = []
        for p in players:
            if p is not None:
                self.race_config.players.append(p.config)
                self.active_players.append(p)
        
        self.k = pystk.Race(self.race_config)

        self.k.start()
        self.k.step()

    def play(self, save=None, max_frames=50, verbose=True):
        # TODO: remove this before submitting
        if verbose:
            # print("hello000000")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1)
        # TODO: end
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
                player_action, guess_coord = p(image, player)
                for a in player_action:
                    setattr(action, a, player_action[a])
                
                list_actions.append(action)

                ball_coords = self.to_numpy(state.soccer.ball.location)
                kart_proj = np.array(state.players[i].camera.projection).T
                kart_view = np.array(state.players[i].camera.view).T
                kart_ball_coords = self._to_image(ball_coords, kart_proj, kart_view)
                # print("actual coords = ", kart_ball_coords)

                if save is not None:
                    # print("porque")
                    PIL.Image.fromarray(image).save(os.path.join(save, 'player%02d_%05d.png' % (i, t)))
                    # TODO: remove this before submitting
                    # ball_location = np.float32(state.soccer.ball.location)
                    # ball_distance = self.get_vector_from_this_to_that(np.float32(state.players[i].kart.location), np.float32(state.soccer.ball.location))
                    # # print("DISTANCCCCEEEE",ball_distance)
                    # with open(save + '/player%02d_%05d.csv' % (i, t), mode='w') as ball_file:
                    #     ball_writer = csv.writer(ball_file, delimiter=',')
                    #     ball_writer.writerow([ball_location[0], ball_location[2]])
                    #     ball_writer.writerow([ball_distance[0], ball_distance[2]])

                if verbose and (i == 0):
                    ax.clear()
                    ax.imshow(self.k.render_data[0].image)
                    WH2 = np.array([self.graphics_config.screen_width, self.graphics_config.screen_height]) / 2
                    ax.add_artist(
                        plt.Circle(WH2 * (1 + self._to_image(state.players[i].kart.location, kart_proj, kart_view)), 2,
                                   ec='b', fill=False, lw=1.5))
                    ax.add_artist(
                        plt.Circle(WH2 * (1 + self._to_image(ball_coords, kart_proj, kart_view)), 2, ec='r', fill=False,
                                   lw=1.5))
                    ax.add_artist(
                        plt.Circle(WH2 * (1 + guess_coord), 2, ec='g', fill=False,
                                   lw=1.5))
                    plt.pause(1e-3)
                # TODO: END
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
