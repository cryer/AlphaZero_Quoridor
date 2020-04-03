import numpy as np
from queue import Queue
import time

from collections import deque

import copy

from constant import BOARD_SIZE, WALL_NUM

HORIZONTAL = 1
VERTICAL = -1


class Quoridor(object):
    HORIZONTAL = 1
    VERTICAL = -1

    def __init__(self, safe=False):
        self.safe = safe

        self.action_space = 12 + (BOARD_SIZE - 1) ** 2  # 12 + 32 actions in total (5x5 Quoridor)
        self.n_players = 2
        self.players = [1, 2]  #
        self.reset()

    # load player's info (human or computer)
    def load(self, p1, p2):
        self.player1 = p1
        self.player2 = p2

    # return current player info
    def get_current_player(self):
        return self.current_player

    # reset and initialize single game
    def reset(self):
        self.current_player = 1
        self.last_player = -1

        # Initialize Tiles
        self.tiles = np.zeros(BOARD_SIZE ** 2)

        # Initialize Player Locations
        self._positions = {
            1: (BOARD_SIZE ** 2) - (BOARD_SIZE//2) - 1,
            2: BOARD_SIZE // 2
        }

        self._DIRECTIONS = {
            'N': 0, 'S': 1, 'E': 2, 'W': 3,
            'NN': 4, 'SS': 5, 'EE': 6, 'WW': 7,
            'NE': 8, 'NW': 9, 'SE': 10, 'SW': 11
        }
        self.N_DIRECTIONS = 12
        self.N_TILES = BOARD_SIZE ** 2
        self.N_ROWS = BOARD_SIZE
        self.N_INTERSECTIONS = (BOARD_SIZE - 1) ** 2

        # There are 16 possible intersection
        # Horizontal Walls - 1
        # No Wall - 0
        # Vertical Wall - -1
        self._intersections = np.zeros((BOARD_SIZE - 1) ** 2)

        self._player_walls_remaining = {1: WALL_NUM, 2: WALL_NUM}

    def state(self):
        """Returns a set of 5x5 planes that represent the game state.
        0. No walls
        1. Vertical walls
        2. Horizontal walls
        3. The current player position
        4. The opponent position
        5 - 8. Number of walls remaining for current player (if 3walls remain, 5/6/8th are zeros and 7th is ones)
        9 - 12. Number of walls remaining for opponent
        13. Whose turn it is (0 for player 1, 1 for player 2)
        """
        player1_position_plane = self.tiles.copy()
        player1_position_plane[self._positions[self.current_player]] = 1
        player1_position_plane = player1_position_plane.reshape([BOARD_SIZE, BOARD_SIZE])

        player2_position_plane = self.tiles.copy()
        player2_position_plane[self._positions[3 - self.current_player]] = 1
        player2_position_plane = player2_position_plane.reshape([BOARD_SIZE, BOARD_SIZE])

        player1_walls_plane = np.zeros([WALL_NUM+1, BOARD_SIZE, BOARD_SIZE])
        player2_walls_plane = np.zeros([WALL_NUM+1, BOARD_SIZE, BOARD_SIZE])

        player1_walls_plane[self._player_walls_remaining[self.current_player], :, :] = 1
        player2_walls_plane[self._player_walls_remaining[3 - self.current_player], :, :] = 1

        # 1 where vertical walls are placed
        # if state size is ?x9x9, use only ?x8x8 field
        vertical_walls = np.pad(
            np.int8(self._intersections == -1).reshape([BOARD_SIZE - 1, BOARD_SIZE - 1]),
            (0, 1),
            mode='constant',
            constant_values=0
        )

        # 1 where horizontal walls are placed
        # if state size is ?x9x9, use only ?x8x8 field
        horizontal_walls = np.pad(
            np.int8(self._intersections == 1).reshape([BOARD_SIZE - 1, BOARD_SIZE - 1]),
            (0, 1),
            mode='constant',
            constant_values=0
        )

        # 1 where any walls exist
        # if state size is ?x9x9, use only ?x8x8 field
        no_walls = np.pad(
            np.int8(self._intersections == 0).reshape([BOARD_SIZE - 1, BOARD_SIZE - 1]),
            (0, 1),
            mode='constant',
            constant_values=0
        )

        # shortest path distance
        dist1, dist2 = self.get_shortest_path()
        dist1_plane = np.zeros(BOARD_SIZE ** 2)
        dist2_plane = np.zeros(BOARD_SIZE ** 2)

        if self.current_player == 1:
            for i in range(dist1):
                dist1_plane[i] = 1
            for i in range(dist2):
                dist2_plane[i] = 1
        else:
            for i in range(dist2):
                dist2_plane[i] = 1
            for i in range(dist1):
                dist1_plane[i] = 1

        dist1_plane = np.reshape(dist1_plane, (1, BOARD_SIZE, BOARD_SIZE))
        dist2_plane = np.reshape(dist2_plane, (1, BOARD_SIZE, BOARD_SIZE))

        state = np.stack([
                no_walls,
                vertical_walls,
                horizontal_walls,
                player1_position_plane,
                player2_position_plane,
            ])

        if self.current_player == 1:
            current_player_plane = np.zeros([1, BOARD_SIZE, BOARD_SIZE])
        else:
            current_player_plane = np.ones([1, BOARD_SIZE, BOARD_SIZE])

        state = np.vstack([state, player1_walls_plane, player2_walls_plane, current_player_plane, dist1_plane, dist2_plane])
        # state = np.vstack([state, player1_walls_plane, player2_walls_plane, current_player_plane])

        return state


    def get_shortest_path(self):

        player1_target = 0
        player2_target = BOARD_SIZE - 1

        player1_position = self._positions[1]
        player2_position = self._positions[2]

        intersections = self._intersections


        player1_valid, dist1 = self._bfs_to_goal2(intersections, player1_target, player1_position, player1_position, player=1)
        player2_valid, dist2 = self._bfs_to_goal2(intersections, player2_target, player2_position, player2_position, player=2)

        return dist1, dist2


    def actions(self):
        player = self.current_player
        location = self._positions[player]

        opponent = 1 if player == 2 else 2
        opponent_loc = self._positions[opponent]
        walls = self._intersections  #
        #
        pawn_actions = self._valid_pawn_actions(location=location,
                                                opponent_loc=opponent_loc, walls=walls, player=player)
        #
        if (self._player_walls_remaining[self.current_player] > 0):
            wall_actions = self._valid_wall_actions()  #

            #
            wall_actions = [action + 12 for action in wall_actions]
        else:
            wall_actions = []


        return pawn_actions + wall_actions

    def step(self, action):
        """Take a step in the environment given the current action"""
        # self._logger.info("Player {player} chooses action {action}".format(player=self.current_player, action=action))
        player = self.current_player
        done = False
        #
        self.valid_actions = self.actions()

        if self.safe:
            if not action in self.valid_actions:
                raise ValueError("Invalid Action: {action}".format(action=action))

        if action < 12:
            self._handle_pawn_action(action, player)
        else:
            self._handle_wall_action(action - 12)

        self.rotate_players()
        """
        game_over, winner = self.has_a_winner()
        if game_over:
            #  print("game over !winner is player" + str(winner))
            done = True
        else:
            self.rotate_players()  #
            # observation = self.get_state  # get_state
            # observation = self.state()
            # print("game over !winner is player" + str(winner))

        return done, winner
        """

    def has_a_winner3(self):
        game_over = False
        winner = None
        if self._positions[2] > (BOARD_SIZE - 1) * BOARD_SIZE - 1:
            winner = 2
            game_over = True
        elif self._positions[1] < BOARD_SIZE:
            winner = 1
            game_over = True
        return game_over, winner

    def has_a_winner(self):
        game_over = False
        winner = None
        dist1, dist2 = self.get_shortest_path()

        if self._positions[2] > (BOARD_SIZE - 1) * BOARD_SIZE - 1:
            winner = 2
            game_over = True
        elif self._positions[1] < BOARD_SIZE:
            winner = 1
            game_over = True

        if abs(dist1 - dist2) > 1 and self._player_walls_remaining[1] == 0 and self._player_walls_remaining[2] == 0:
            winner = 2 if dist1 > dist2 else 1
            game_over = True

        return game_over, winner


    # check that pawn moved to dead end
    def is_dead_end(self, game):
        current_player = self.get_current_player()



    def _return_pawn_action(self, action, player):
        if action == self._DIRECTIONS['N']:
            return self._positions[player] - BOARD_SIZE
        elif action == self._DIRECTIONS['S']:
            return self._positions[player] + BOARD_SIZE
        elif action == self._DIRECTIONS['E']:
            return self._positions[player] + 1
        elif action == self._DIRECTIONS['W']:
            return self._positions[player] - 1
        elif action == self._DIRECTIONS['NN']:
            return self._positions[player] - (BOARD_SIZE * 2)
        elif action == self._DIRECTIONS['SS']:
            return self._positions[player] + (BOARD_SIZE * 2)
        elif action == self._DIRECTIONS['EE']:
            return self._positions[player] + 2
        elif action == self._DIRECTIONS['WW']:
            return self._positions[player] - 2
        elif action == self._DIRECTIONS['NW']:
            return self._positions[player] - (BOARD_SIZE + 1)
        elif action == self._DIRECTIONS['NE']:
            return self._positions[player] - (BOARD_SIZE - 1)
        elif action == self._DIRECTIONS['SW']:
            return self._positions[player] + (BOARD_SIZE - 1)
        elif action == self._DIRECTIONS['SE']:
            return self._positions[player] + (BOARD_SIZE + 1)
        else:
            raise ValueError("Invalid Pawn Action: {action}".format(action=action))


    def _handle_pawn_action(self, action, player):
        if action == self._DIRECTIONS['N']:
            self._positions[player] -= BOARD_SIZE
        elif action == self._DIRECTIONS['S']:
            self._positions[player] += BOARD_SIZE
        elif action == self._DIRECTIONS['E']:
            self._positions[player] += 1
        elif action == self._DIRECTIONS['W']:
            self._positions[player] -= 1
        elif action == self._DIRECTIONS['NN']:
            self._positions[player] -= BOARD_SIZE * 2
        elif action == self._DIRECTIONS['SS']:
            self._positions[player] += BOARD_SIZE *  2
        elif action == self._DIRECTIONS['EE']:
            self._positions[player] += 2
        elif action == self._DIRECTIONS['WW']:
            self._positions[player] -= 2
        elif action == self._DIRECTIONS['NW']:
            self._positions[player] -= BOARD_SIZE + 1
        elif action == self._DIRECTIONS['NE']:
            self._positions[player] -= BOARD_SIZE - 1
        elif action == self._DIRECTIONS['SW']:
            self._positions[player] += BOARD_SIZE - 1
        elif action == self._DIRECTIONS['SE']:
            self._positions[player] += BOARD_SIZE + 1
        else:
            raise ValueError("Invalid Pawn Action: {action}".format(action=action))

    def _handle_wall_action(self, action):
        # Action values less than 16 are horizontal walls
        if action < (BOARD_SIZE - 1) ** 2:
            self._intersections[action] = 1
        # Action values above 16 are vertical walls
        else:
            self._intersections[action - (BOARD_SIZE - 1) ** 2] = -1

        self._player_walls_remaining[self.current_player] -= 1
        # self._logger.info(self._intersections)

    def rotate_players(self):
        """Switch the player turn"""
        # self._logger.debug("Rotating Player")
        if self.current_player == 1:
            self.current_player = 2
            self.last_player = 1

        else:
            self.current_player = 1
            self.last_player = 2

    def _valid_pawn_actions(self, walls, location, opponent_loc, player=1):

        valid = []
        opponent_north = (location == opponent_loc + BOARD_SIZE)
        opponent_south = (location == opponent_loc - BOARD_SIZE)
        opponent_east = (location == opponent_loc - 1)
        opponent_west = (location == opponent_loc + 1)

        current_row = location // self.N_ROWS

        intersections = self._get_intersections(walls, location)
        n = intersections['NW'] != HORIZONTAL and intersections['NE'] != HORIZONTAL and not opponent_north
        s = intersections['SW'] != HORIZONTAL and intersections['SE'] != HORIZONTAL and not opponent_south
        e = intersections['NE'] != VERTICAL and intersections['SE'] != VERTICAL and not opponent_east
        w = intersections['NW'] != VERTICAL and intersections['SW'] != VERTICAL and not opponent_west
        if n: valid.append(self._DIRECTIONS['N'])
        if s: valid.append(self._DIRECTIONS['S'])
        if e: valid.append(self._DIRECTIONS['E'])
        if w: valid.append(self._DIRECTIONS['W'])


        if opponent_north and intersections['NE'] != HORIZONTAL and intersections['NW'] != HORIZONTAL:
            n_intersections = self._get_intersections(walls, opponent_loc)
            if n_intersections['NW'] != HORIZONTAL and n_intersections['NE'] != HORIZONTAL:  # or (current_row == 3 and player == 1):
                valid.append(self._DIRECTIONS['NN'])

            if n_intersections['NE'] != VERTICAL and intersections['NE'] != VERTICAL:
                valid.append(self._DIRECTIONS['NE'])

            if n_intersections['NW'] != VERTICAL and intersections['NW'] != VERTICAL:
                valid.append(self._DIRECTIONS['NW'])


        elif opponent_south and intersections['SE'] != HORIZONTAL and intersections['SW'] != HORIZONTAL:
            s_intersections = self._get_intersections(walls, opponent_loc)
            if s_intersections['SW'] != HORIZONTAL and s_intersections['SE'] != HORIZONTAL:  # or (current_row == 1 and player == 2):
                valid.append(self._DIRECTIONS['SS'])

            if s_intersections['SE'] != VERTICAL and intersections['SE'] != VERTICAL:
                valid.append(self._DIRECTIONS['SE'])

            if s_intersections['SW'] != VERTICAL and intersections['SW'] != VERTICAL:
                valid.append(self._DIRECTIONS['SW'])


        elif opponent_east and intersections['SE'] != VERTICAL and intersections['NE'] != VERTICAL:
            e_intersections = self._get_intersections(walls, opponent_loc)
            if e_intersections['SE'] != VERTICAL and e_intersections['NE'] != VERTICAL:
                valid.append(self._DIRECTIONS['EE'])

            # fix jumping over wall bug
            if e_intersections['NE'] != HORIZONTAL and e_intersections['NW'] != HORIZONTAL:
                valid.append(self._DIRECTIONS['NE'])

            # fix jumping over wall bug
            if e_intersections['SE'] != HORIZONTAL and e_intersections['SW'] != HORIZONTAL:
                valid.append(self._DIRECTIONS['SE'])


        elif opponent_west and intersections['SW'] != VERTICAL and intersections['NW'] != VERTICAL:
            w_intersections = self._get_intersections(walls, opponent_loc)
            if w_intersections['NW'] != VERTICAL and w_intersections['SW'] != VERTICAL:
                valid.append(self._DIRECTIONS['WW'])

            # fix jumping over wall bug
            if w_intersections['NW'] != HORIZONTAL and w_intersections['NE'] != HORIZONTAL:
                valid.append(self._DIRECTIONS['NW'])

            # fix jumping over wall bug
            if w_intersections['SW'] != HORIZONTAL and w_intersections['SE'] != HORIZONTAL:
                valid.append(self._DIRECTIONS['SW'])

        return valid

    def _get_intersections(self, intersections, current_tile):


        """Gets the four intersections for a given tile."""
        location_row = current_tile // self.N_ROWS
        s_border = current_tile > (BOARD_SIZE - 1) * BOARD_SIZE -1
        e_border = current_tile % BOARD_SIZE == BOARD_SIZE - 1
        n_border = current_tile < BOARD_SIZE
        w_border = current_tile % BOARD_SIZE == 0

        if n_border:
            ne_intersection = HORIZONTAL

            location_row = 0
            if w_border:
                nw_intersection = VERTICAL
                sw_intersection = VERTICAL
                se_intersection = intersections[current_tile]
            elif e_border:
                nw_intersection = HORIZONTAL
                se_intersection = VERTICAL
                sw_intersection = intersections[current_tile - 1]
            else:
                nw_intersection = HORIZONTAL
                sw_intersection = intersections[current_tile - 1]
                se_intersection = intersections[current_tile]
        elif s_border:
            location_row = BOARD_SIZE - 1
            sw_intersection = HORIZONTAL
            if w_border:
                nw_intersection = VERTICAL
                se_intersection = HORIZONTAL
                ne_intersection = intersections[(current_tile- BOARD_SIZE) - (location_row-1)]
            elif e_border:
                se_intersection = VERTICAL
                ne_intersection = VERTICAL
                nw_intersection = intersections[(current_tile - BOARD_SIZE) - location_row]
            else:
                se_intersection = HORIZONTAL
                ne_intersection = intersections[(current_tile - BOARD_SIZE) - (location_row -1)]
                nw_intersection = intersections[(current_tile - BOARD_SIZE) - location_row]


        # West but not north or south
        elif w_border:
            nw_intersection = VERTICAL
            sw_intersection = VERTICAL
            se_intersection = intersections[current_tile - location_row]
            ne_intersection = intersections[(current_tile - BOARD_SIZE) - (location_row - 1)]

        elif e_border:
            ne_intersection = VERTICAL
            se_intersection = VERTICAL
            sw_intersection = intersections[current_tile - location_row - 1]
            nw_intersection = intersections[(current_tile - BOARD_SIZE) - (location_row - 1) - 1]

        # No borders
        else:
            ne_intersection = intersections[(current_tile - BOARD_SIZE) - (location_row-1)]
            nw_intersection = intersections[(current_tile - BOARD_SIZE) - location_row ]
            sw_intersection = intersections[current_tile - (location_row +1)]
            se_intersection = intersections[current_tile - location_row]

        return {'NW': nw_intersection,
                'NE': ne_intersection,
                'SE': se_intersection,
                'SW': sw_intersection}

    def _valid_wall_actions(self):
        valid = []
        # If
        for ix in range(self._intersections.size):
            if self._validate_horizontal(ix):
                valid.append(ix)

            if self._validate_vertical(ix):
                valid.append(ix + (BOARD_SIZE - 1) ** 2)

        return valid

    def _validate_horizontal(self, ix):
        column = ix % (BOARD_SIZE - 1)

        if self._intersections[ix] != 0:
            return False

        if column != 0:
            if self._intersections[ix - 1] == 1:
                return False

        if column != BOARD_SIZE - 2:
            if self._intersections[ix + 1] == 1:
                return False

        return not self._blocks_path(ix, self.HORIZONTAL)

    def _validate_vertical(self, ix):
        row = ix // (BOARD_SIZE - 1)
        if self._intersections[ix] != 0:
            return False

        if row != 0:
            if self._intersections[ix - (BOARD_SIZE - 1)] == -1:
                return False

        if row != BOARD_SIZE - 2:
            if self._intersections[ix + (BOARD_SIZE - 1)] == -1:
                return False

        return not self._blocks_path(ix, self.VERTICAL)

    def _blocks_path(self, wall_location, orientation):
        player1_target = 0
        player2_target = BOARD_SIZE - 1

        player1_position = self._positions[1]
        player2_position = self._positions[2]

        intersections = self._intersections.copy()
        intersections[wall_location] = orientation

        # BFS to target row
        player1_valid = self._bfs_to_goal(intersections, player1_target, player1_position, player2_position, player=1)
        player2_valid = self._bfs_to_goal(intersections, player2_target, player2_position, player1_position, player=2)

        return not (player1_valid and player2_valid)

    def _bfs_to_goal(self, intersections, target_row, player_position, opponent_position, player=1):
        visited = []
        invalid_rows = [BOARD_SIZE, -1]
        visit_queue = Queue()
        visit_queue.put(player_position)
        target_visited = False


        while not target_visited and not visit_queue.empty():
            current_position = visit_queue.get()
            valid_directions = self._valid_pawn_actions(intersections,
                                                        location=current_position,
                                                        opponent_loc=opponent_position,
                                                        player=player)

            for direction in valid_directions:
                if direction == self._DIRECTIONS['N']:
                    new_position = current_position - BOARD_SIZE
                elif direction == self._DIRECTIONS['S']:
                    new_position = current_position + BOARD_SIZE
                elif direction == self._DIRECTIONS['E']:
                    new_position = current_position + 1
                elif direction == self._DIRECTIONS['W']:
                    new_position = current_position - 1
                elif direction == self._DIRECTIONS['NN']:
                    new_position = current_position - (BOARD_SIZE * 2)
                elif direction == self._DIRECTIONS['SS']:
                    new_position = current_position + (BOARD_SIZE * 2)
                elif direction == self._DIRECTIONS['EE']:
                    new_position = current_position + 2
                elif direction == self._DIRECTIONS['WW']:
                    new_position = current_position - 2
                elif direction == self._DIRECTIONS['NE']:
                    new_position = current_position - (BOARD_SIZE - 1)
                elif direction == self._DIRECTIONS['NW']:
                    new_position = current_position - (BOARD_SIZE + 1)
                elif direction == self._DIRECTIONS['SW']:
                    new_position = current_position + (BOARD_SIZE - 1)
                elif direction == self._DIRECTIONS['SE']:
                    new_position = current_position + (BOARD_SIZE + 1)
                else:
                    raise ValueError('Invalid direction - should never happen')


                new_row = new_position // self.N_ROWS
                if new_row == target_row:
                    target_visited = True
                elif new_position not in visited:
                    visited.append(new_position)
                    if new_row not in invalid_rows:
                        visit_queue.put(new_position)


        return target_visited

    def _bfs_to_goal2(self, intersections, target_row, player_position, opponent_position, player=1):
        visited = []
        invalid_rows = [BOARD_SIZE, -1]
        visit_queue = Queue()
        visit_queue.put(player_position)
        target_visited = False

        dist = 0

        temp_queue = Queue()

        while not target_visited and not visit_queue.empty():

            while not visit_queue.empty() :
                current_position = visit_queue.get()

                if current_position < 0 or current_position > (BOARD_SIZE ** 2 -1):
                    print("Strange position in the queue: ", current_position)

                valid_directions = self._valid_pawn_actions(intersections,
                                                        location=current_position,
                                                        opponent_loc=opponent_position,
                                                        player=player)


                for direction in valid_directions:
                    if direction == self._DIRECTIONS['N']:
                        new_position = current_position - BOARD_SIZE
                    elif direction == self._DIRECTIONS['S']:
                        new_position = current_position + BOARD_SIZE
                    elif direction == self._DIRECTIONS['E']:
                        new_position = current_position + 1
                    elif direction == self._DIRECTIONS['W']:
                        new_position = current_position - 1
                    elif direction == self._DIRECTIONS['NN']:
                        new_position = current_position - (BOARD_SIZE * 2)
                    elif direction == self._DIRECTIONS['SS']:
                        new_position = current_position + (BOARD_SIZE * 2)
                    elif direction == self._DIRECTIONS['EE']:
                        new_position = current_position + 2
                    elif direction == self._DIRECTIONS['WW']:
                        new_position = current_position - 2
                    elif direction == self._DIRECTIONS['NE']:
                        new_position = current_position - (BOARD_SIZE - 1)
                    elif direction == self._DIRECTIONS['NW']:
                        new_position = current_position - (BOARD_SIZE + 1)
                    elif direction == self._DIRECTIONS['SW']:
                        new_position = current_position + (BOARD_SIZE - 1)
                    elif direction == self._DIRECTIONS['SE']:
                        new_position = current_position + (BOARD_SIZE + 1)
                    else:
                        raise ValueError('Invalid direction - should never happen')

                    new_row = new_position // self.N_ROWS
                    if new_row == target_row:
                        return True, dist + 1
                    elif new_position not in visited:
                        visited.append(new_position)
                        temp_queue.put(new_position)

            while not temp_queue.empty():
                position = temp_queue.get()
                if position < 0 or position > (BOARD_SIZE ** 2 - 1):
                    continue
                else:
                    visit_queue.put(position)

            dist += 1

        return target_visited, dist



    def add_wall(self, wall, orientation):
        self._intersections[wall] = orientation

    def print_board(self):
        player1_row = self._positions[1] // BOARD_SIZE * 2
        player1_col = self._positions[1] % BOARD_SIZE * 2
        player2_row = self._positions[2] // BOARD_SIZE * 2
        player2_col = self._positions[2] % BOARD_SIZE * 2

        x = 'X'
        o = 'O'

        v = 'v'
        h = 'h'
        dash = '-'
        none = ''

        grid_new = np.zeros([BOARD_SIZE * 2 - 1, BOARD_SIZE * 2 - 1])
        grid_new[player1_row, player1_col] = 1
        grid_new[player2_row, player2_col] = 2

        i_reshaped = self._intersections.reshape([BOARD_SIZE - 1, BOARD_SIZE - 1])
        for i in range(BOARD_SIZE -1):
            for j in range(BOARD_SIZE - 1):
                if i_reshaped[i][j] == 1:
                    grid_new[((i * 2) + 1), (j * 2):((j * 2) + 3)] = 3
                elif i_reshaped[i][j] == -1:
                    grid_new[(i * 2):((i * 2) + 3), ((j * 2) + 1)] = 4

        for i in range(BOARD_SIZE * 2 - 1):
            render_row = ""
            for j in range(BOARD_SIZE * 2 - 1):
                if grid_new[i, j] == 0:
                    if i % 2 == 0 and j % 2 == 0:
                        render_row += " . "
                    else:
                        render_row += "   "
                elif grid_new[i, j] == 1:
                    render_row += " X "
                elif grid_new[i, j] == 2:
                    render_row += " O "
                elif grid_new[i, j] == 3:
                    render_row += "---"
                else:
                    render_row += " | "

            print(render_row)

        """
        # Original print board code

        grid = [['{dash:4}'.format(dash=dash) for i in range(5)] for i in range(5)]
        i_reshaped = self._intersections.reshape([4, 4])

        grid[player1_row][player1_col] = '{x:4}'.format(x=x)
        grid[player2_row][player2_col] = '{o:4}'.format(o=o)

        intersection_row = 3
        for i in range(4, -1, -1):
            for j in range(5):
                print(grid[i][j], end='')
            print()
            if intersection_row >= 0:
                print('{none:2}'.format(none=none), end='')
                for j in i_reshaped[intersection_row, :]:
                    if j == 1:
                        # print('{h:4}'.format(h=h), end='')
                        print('{h:4}'.format(h=h), end='')
                    elif j == -1:
                        print('{v:4}'.format(v=v), end='')
                    else:
                        print('{none:4}'.format(none=none), end='')
                intersection_row -= 1
                print()
        """

    def clone(self):
        return Quoridor()

    def start_self_play(self, player, is_shown=0, temp=1e-3):

        self.reset()     #
        p1, p2 = self.players
        states, mcts_probs, current_players = [], [], []       #

        time_step = 0


        while(1):

            time_step += 1

            tic = time.time()
            move, move_probs = player.choose_action(self, temp=temp, return_prob=1, time_step=time_step)
            toc = time.time()
            print('[Move probs]\n', move_probs[:12])
            print('[Wall probs]\n', move_probs[12:])
            print("player %s chose move : %s, prob: %.3f, spend: %.2f seconds" % (self.current_player, move, move_probs[move], (toc-tic)))


            states.append(self.state())
            mcts_probs.append(move_probs)
            current_players.append(self.current_player)

            self.step(move)

            dist1, dist2 = self.get_shortest_path()
            print("Player 1 Shortest Path: {}, Player 2 Shortest Path: {}".format(dist1, dist2))


            self.print_board()
            # if is_shown:
            #     self.graphic(self.board, p1, p2)
            end, winner = self.has_a_winner()

            if end:
                #
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0        #
                    winners_z[np.array(current_players) != winner] = -1.0       #
                #
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)
