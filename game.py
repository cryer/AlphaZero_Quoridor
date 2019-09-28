import pygame
from quoridor import Quoridor
from agents.base import BaseAgent
from agents.manual import ManualPygameAgent
from mcts import MCTSPlayer as A_Player
from pure_mcts import MCTSPlayer as B_Player
from policy_value_net import PolicyValueNet
# add
import numpy as np
import time
import argparse

# Define Colors
BLACK = (0, 0, 0)
WHITE = (240, 255, 240)
LIGHTBROWN = (222, 184, 135)
BROWN = (128, 0, 0)
LIGHTRED = (240, 128, 128)
RED = (205, 92, 92)
LIGHTBLUE = (221, 160, 221)
BLUE = (186, 85, 211)
DARKBLUE = (0, 0, 128)

SCREEN_WIDTH = 600
SCREEN_HEIGHT = SCREEN_WIDTH - 200

TILE_WIDTH = SCREEN_HEIGHT / 10.6
TILE_HEIGHT = SCREEN_HEIGHT / 10.6

WALL_WIDTH = 0.2 * TILE_WIDTH
WALL_HEIGHT = TILE_WIDTH * 2 + WALL_WIDTH


def render(game, screen):
    valid_actions = game.actions()
    draw_game(game, screen, valid_actions)


def text(screen, text, position1=2, position2=0.6, color=BLUE):
    font = pygame.font.SysFont("arial", 18)
    a = font.render(
        "%s" % text, 1, color)
    p = SCREEN_HEIGHT + position1, SCREEN_HEIGHT * position2
    screen.blit(a, p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--player_type", type=int, default=1,
                        help="palyer type you want to fight,1 is human,2 is computer")
    parser.add_argument("--computer_type", type=int, default=0, help="computer type,1 is Alpha MCTS,2 is pure MCTS")
    args = parser.parse_args()

    game = Quoridor()
    human1 = ManualPygameAgent('Kurumi')
    human2 = ManualPygameAgent('Cryer')
    MCTS_Alpha = A_Player(PolicyValueNet().policy_value_fn, c_puct=5, n_playout=30, is_selfplay=0)
    MCTS_Pure = B_Player(c_puct=5, n_playout=50)  # 50层400秒

    if args.player_type == 1:
        player_types = {1: 'human', 2: 'human'}
        players = {1: human1, 2: human2}
        if args.computer_type == 0:
            pass
    elif args.player_type == 2:
        player_types = {1: 'human', 2: 'computer'}
        if args.computer_type == 1:
            players = {1: human1, 2: MCTS_Alpha}
        elif args.computer_type == 2:
            players = {1: human1, 2: MCTS_Pure}
        elif args.computer_type == 0:
            print("Set computer type to 1 or 2 for choosing computer!")
            # pygame.quit()

    # game.load(player1, player2)

    pygame.init()

    WINDOW_SIZE = [SCREEN_WIDTH, SCREEN_HEIGHT]
    screen = pygame.display.set_mode(WINDOW_SIZE)

    pygame.display.set_caption("QUORIDOR")

    clock = pygame.time.Clock()

    # valid_actions = game.valid_actions  11
    valid_actions = game.actions()
    done = False
    winner = None
    t1 = time.time()
    while not done:
        player_moved = False

        # 定义落子历史
        # move_history = []

        pawn_moves, walls = draw_game(game, screen, valid_actions)

        # text(screen, "player1 move:", position1=2, position2=0.8, color=BLUE)

        valid_walls = [wall for wall in walls if wall[2] in valid_actions]
        if player_types[game.current_player] == 'human':
            touch = pygame.mouse.get_pos()
            for wall, collides, _ in valid_walls:
                for collide in collides:
                    if collide.collidepoint(touch):
                        pygame.draw.rect(screen, LIGHTBROWN, wall)
                        break
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    touch = pygame.mouse.get_pos()
                    # This is messy - fix later
                    for rect, action in pawn_moves:
                        if rect.collidepoint(touch):
                            players[game.current_player].receive_action(action)
                            player_moved = True
                            break
                        if player_moved: break
                    # if player_moved: break
                    # 添加
                    if player_moved:
                        real_action = players[game.current_player].choose_action()
                        # move_history.append(real_action)
                        done, winner = game.step(real_action)
                        render(game, screen)  # 渲染游戏
                        break

                    for rect, collide_points, action in valid_walls:
                        for collides in collide_points:
                            if collides.collidepoint(touch):
                                players[game.current_player].receive_action(action)
                                player_moved = True
                                break
                        # 修改
                        if player_moved == True:
                            real_action = players[game.current_player].choose_action()
                            # move_history.append(real_action)
                            done, winner = game.step(real_action)
                            render(game, screen)  # 渲染游戏
                            break

        clock.tick(30)
        pygame.display.flip()

        valid_actions = game.actions()

        # 待改
        if player_types[game.current_player] == 'computer':
            print("computer %s thinking..." % str(game.current_player))
            tic = time.time()
            # real_action = np.random.choice(valid_actions)
            real_action = players[game.current_player].choose_action(game)
            # move_history.append(real_action)
            toc = time.time()
            print("MCTS choose action:", real_action, "  ,spend %s seconds" % str(toc - tic))
            done, winner = game.step(real_action)
            # render(game, screen)
            # valid_actions = game.valid_actions
        # if game.current_player == 1:
        #     text(screen, text, position1=2, position2=0.8, color=BLUE)

        if done:
            print("game over! winner is %s player:%s" % (player_types[winner], winner))
            break

    t2 = time.time()
    print("total time :", t2 - t1)
    pygame.quit()


def draw_game(game, screen, valid_actions):
    # Calculate valid action tiles
    # Draw Valid Pawn actions
    screen.fill(BLACK)
    pawn_actions = [action for action in valid_actions if action < 12]
    reference_tile = game._positions[game.current_player]  # 当前玩家的位置，一个标量

    action_tiles = {}
    # 遍历所有合法的棋子移动，action是一个int
    for action in pawn_actions:
        if action == game._DIRECTIONS['N']:
            action_tiles[reference_tile + 9] = action
        elif action == game._DIRECTIONS['S']:
            action_tiles[reference_tile - 9] = action
        elif action == game._DIRECTIONS['E']:
            action_tiles[reference_tile + 1] = action
        elif action == game._DIRECTIONS['W']:
            action_tiles[reference_tile - 1] = action
        elif action == game._DIRECTIONS['NN']:
            action_tiles[reference_tile + 18] = action
        elif action == game._DIRECTIONS['SS']:
            action_tiles[reference_tile - 18] = action
        elif action == game._DIRECTIONS['EE']:
            action_tiles[reference_tile + 2] = action
        elif action == game._DIRECTIONS['WW']:
            action_tiles[reference_tile - 2] = action
        elif action == game._DIRECTIONS['NE']:
            action_tiles[reference_tile + 10] = action
        elif action == game._DIRECTIONS['NW']:
            action_tiles[reference_tile + 8] = action
        elif action == game._DIRECTIONS['SE']:
            action_tiles[reference_tile - 8] = action
        elif action == game._DIRECTIONS['SW']:
            action_tiles[reference_tile - 10] = action
    # action_tiles key是位置，value是上一步动作action的值0-11
    # Draw Tiles
    pawn_moves = []
    for row in range(9):
        for column in range(9):
            if row * 9 + column in action_tiles.keys():
                if game.current_player == 1:
                    color = LIGHTBLUE
                else:
                    color = LIGHTRED
                rect = pygame.draw.rect(
                    screen,
                    color,
                    [(TILE_WIDTH + WALL_WIDTH) * column,
                     (WALL_WIDTH + TILE_HEIGHT) * (8 - row),
                     TILE_WIDTH,
                     TILE_HEIGHT]
                )
                pawn_moves.append([rect, action_tiles[row * 9 + column]])
            else:
                if row * 9 + column == game._positions[1]:
                    color = BLUE
                elif row * 9 + column == game._positions[2]:
                    color = RED
                else:
                    color = DARKBLUE
                pygame.draw.rect(screen,
                                 color,
                                 [(TILE_WIDTH + WALL_WIDTH) * column,
                                  (WALL_WIDTH + TILE_HEIGHT) * (8 - row),
                                  TILE_WIDTH,
                                  TILE_HEIGHT])

    walls = []

    # Draw Vertical Walls
    placed_walls = []
    for row in range(8):
        for column in range(8):
            collide_points = []
            rect = pygame.Rect(TILE_WIDTH + (TILE_WIDTH + WALL_WIDTH) * column,
                               (TILE_HEIGHT + WALL_WIDTH) * (7 - row),
                               WALL_WIDTH,
                               WALL_HEIGHT)
            if game._intersections[row * 8 + column] == -1:
                placed_walls.append(rect)
            else:
                # Collide rectangles for highlighting the walls on hover
                collide_top = pygame.Rect(TILE_WIDTH + (TILE_WIDTH + WALL_WIDTH) * column,
                                          (TILE_HEIGHT + WALL_WIDTH) * (7 - row) + TILE_HEIGHT / 2,
                                          WALL_WIDTH,
                                          TILE_HEIGHT / 2)
                pygame.draw.rect(screen, BLACK, collide_top)
                collide_points.append(collide_top)

                collide_bottom = pygame.Rect(TILE_WIDTH + (TILE_WIDTH + WALL_WIDTH) * column,
                                             (TILE_HEIGHT + WALL_WIDTH) * (7 - row) + TILE_HEIGHT + WALL_WIDTH,
                                             WALL_WIDTH,
                                             TILE_HEIGHT / 2)
                pygame.draw.rect(screen, BLACK, collide_bottom)
                collide_points.append(collide_bottom)

            pygame.draw.rect(screen, BLACK, rect)
            walls.append([rect, collide_points, row * 8 + column + 64 + 12])

    # Draw Horizontal Walls
    for row in range(8):
        for column in range(8):
            rect = pygame.Rect((TILE_HEIGHT + WALL_WIDTH) * column,
                               TILE_HEIGHT + (TILE_HEIGHT + WALL_WIDTH) * (7 - row),
                               WALL_HEIGHT,
                               WALL_WIDTH)
            if game._intersections[row * 8 + column] == 1:
                placed_walls.append(rect)
            else:
                # Collide rectangles for highlighting the walls on hover
                collide_points = []

                collide_left = pygame.Rect((TILE_HEIGHT + WALL_WIDTH) * column + TILE_WIDTH / 2,
                                           TILE_HEIGHT + (TILE_HEIGHT + WALL_WIDTH) * (7 - row),
                                           TILE_WIDTH / 2,
                                           WALL_WIDTH)

                pygame.draw.rect(screen, BLACK, collide_left)
                collide_points.append(collide_left)

                collide_right = pygame.Rect((TILE_HEIGHT + WALL_WIDTH) * column + TILE_WIDTH + WALL_WIDTH,
                                            TILE_HEIGHT + (TILE_HEIGHT + WALL_WIDTH) * (7 - row),
                                            TILE_WIDTH / 2,
                                            WALL_WIDTH)
                pygame.draw.rect(screen, BLACK, collide_right)
                collide_points.append(collide_right)

            rect = pygame.Rect((TILE_HEIGHT + WALL_WIDTH) * (column),
                               TILE_HEIGHT + (TILE_HEIGHT + WALL_WIDTH) * (7 - row),
                               WALL_HEIGHT,
                               WALL_WIDTH)

            pygame.draw.rect(screen, BLACK, rect)
            walls.append([rect, collide_points, row * 8 + column + 12])

    for wall in placed_walls:
        pygame.draw.rect(screen, BROWN, wall)

    # Draw Walls Remaining
    font = pygame.font.SysFont("arial", 18)

    player1_walls = font.render("Walls Remaining: {player1}".format(player1=game._player1_walls_remaining), 1, BLUE)
    player1_text_position = SCREEN_HEIGHT + 2, SCREEN_HEIGHT * 0.9

    player2_walls = font.render("Walls Remaining: {player2}".format(player2=game._player2_walls_remaining), 1, RED)
    player2_text_position = SCREEN_HEIGHT + 2, SCREEN_HEIGHT * 0.1

    move_history = []

    screen.blit(player1_walls, player1_text_position)
    screen.blit(player2_walls, player2_text_position)
    return pawn_moves, walls


def draw_load_screen(screen):
    menu_data = (
        'Player1',
        'Human',
        'AI'
    )

    menu_data = (
        'Player2',
        'Human',
        'AI'
    )


if __name__ == '__main__':
    main()

