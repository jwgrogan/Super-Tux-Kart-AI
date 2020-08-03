from .utils import Player
from .utils import Tournament
from argparse import ArgumentParser
import importlib

if __name__ == '__main__':
    parser = ArgumentParser("Play some Ice Hockey. List any number of players, odd players are in team 1, even players team 2.")
    parser.add_argument('-s', '--save_loc', help="Do you want to record?")
    parser.add_argument('-f', '--num_frames', default=1000, type=int, help="How many steps should we play for?")
    parser.add_argument('players', nargs='+', help="Add any number of players. List python module names or `AI` for AI players). Teams alternate.")
    args = parser.parse_args()
    
    players = []
    for i, player in enumerate(args.players):
        if player == 'AI':
            players.append(None)
        else:
            players.append(Player(importlib.import_module(player).HockeyPlayer(i), i % 2))
        
    tournament = Tournament(players)
    score = tournament.play(save=args.save_loc, max_frames=args.num_frames)
    tournament.close()
    print('Final score', score)
