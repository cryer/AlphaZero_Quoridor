import numpy as np
from queue import Queue
import time

class Quoridor(object):
    HORIZONTAL = 1
    VERTICAL = -1

    def __init__(self, safe=False):
        self.safe = safe

        self.action_space = 44  # 12 + 32 actions in total (5x5 Quoridor)
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
        self.tiles = np.zeros(25)

        # Initialize Player Locations
        self._positions = {
            1: 2,
            2: 22
        }

        self._DIRECTIONS = {
            'N': 0, 'S': 1, 'E': 2, 'W': 3,
            'NN': 4, 'SS': 5, 'EE': 6, 'WW': 7,
            'NE': 8, 'NW': 9, 'SE': 10, 'SW': 11
        }
        self.N_DIRECTIONS = 12
        self.N_TILES = 25
        self.N_ROWS = 5
        self.N_INTERSECTIONS = 16

        # There are 16 possible intersection
        # Horizontal Walls - 1
        # No Wall - 0
        # Vertical Wall - -1
        self._intersections = np.zeros(16)

        self._player1_walls_remaining = 3
        self._player2_walls_remaining = 3

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
        player1_position_plane[self._positions[1]] = 1
        player1_position_plane = player1_position_plane.reshape([5, 5])

        player2_position_plane = self.tiles.copy()
        player2_position_plane[self._positions[2]] = 1
        player2_position_plane = player2_position_plane.reshape([5, 5])

        player1_walls_plane = np.zeros([3, 5, 5])
        player2_walls_plane = np.zeros([3, 5, 5])

        player1_walls_plane[self._player1_walls_remaining - 1, :, :] = 1
        player2_walls_plane[self._player2_walls_remaining - 1, :, :] = 1

        # 1 where vertical walls are placed
        # if state size is ?x9x9, use only ?x8x8 field
        vertical_walls = np.pad(
            np.int8(self._intersections == -1).reshape([4, 4]),
            (0, 1),
            mode='constant',
            constant_values=0
        )

        # 1 where horizontal walls are placed
        # if state size is ?x9x9, use only ?x8x8 field
        horizontal_walls = np.pad(
            np.int8(self._intersections == 1).reshape([4, 4]),
            (0, 1),
            mode='constant',
            constant_values=0
        )

        # 1 where any walls exist
        # if state size is ?x9x9, use only ?x8x8 field
        no_walls = np.pad(
            np.int8(self._intersections == 0).reshape([4, 4]),
            (0, 1),
            mode='constant',
            constant_values=0
        )

        # stack
        if self.current_player == 1:
            state = np.stack([
                no_walls,
                vertical_walls,
                horizontal_walls,
                player1_position_plane,
                player2_position_plane,
            ])

            # print('Shape is {shape}'.format(shape=state.shape))

            current_player_plane = np.zeros([1, 5, 5])
            state = np.vstack([state, player1_walls_plane, player2_walls_plane, current_player_plane])

        if self.current_player == 2:
            state = np.stack([
                no_walls,
                vertical_walls,
                horizontal_walls,
                player2_position_plane,
                player1_position_plane,
            ])

            current_player_plane = np.ones([1, 5, 5])
            state = np.vstack([state, player2_walls_plane, player1_walls_plane, current_player_plane])
            # print(state.shape)
        return state

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
        if ((self.current_player == 1 and self._player1_walls_remaining > 0)
            or (self.current_player == 2 and self._player2_walls_remaining > 0)):
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

    # 
    def has_a_winner(self):
        game_over = False
        winner = None
        if self._positions[2] < 5:
            winner = 2
            game_over = True
        elif self._positions[1] > 19:
            winner = 1
            game_over = True
        return game_over, winner
    

    # check that pawn moved to dead end
    def is_dead_end(self, game):
        current_player = self.get_current_player()


    def _get_rewards(self):
        done = True
        if self._positions[2] < 5:
            rewards, done = (1, -1)
        elif self._positions[1] > 19:
            rewards = (-1, 1)
        else:
            rewards = (0, 0)
            done = False
        return rewards, done

    def _handle_pawn_action(self, action, player):
        if action == self._DIRECTIONS['N']:
            self._positions[player] += 5
        elif action == self._DIRECTIONS['S']:
            self._positions[player] -= 5
        elif action == self._DIRECTIONS['E']:
            self._positions[player] += 1
        elif action == self._DIRECTIONS['W']:
            self._positions[player] -= 1
        elif action == self._DIRECTIONS['NN']:
            self._positions[player] += 10
        elif action == self._DIRECTIONS['SS']:
            self._positions[player] -= 10
        elif action == self._DIRECTIONS['EE']:
            self._positions[player] += 2
        elif action == self._DIRECTIONS['WW']:
            self._positions[player] -= 2
        elif action == self._DIRECTIONS['NW']:
            self._positions[player] += 4
        elif action == self._DIRECTIONS['NE']:
            self._positions[player] += 6
        elif action == self._DIRECTIONS['SW']:
            self._positions[player] -= 6
        elif action == self._DIRECTIONS['SE']:
            self._positions[player] -= 4
        else:
            raise ValueError("Invalid Pawn Action: {action}".format(action=action))

    def _handle_wall_action(self, action):
        # Action values less than 16 are horizontal walls
        if action < 16:
            self._intersections[action] = 1
        # Action values above 16 are vertical walls
        else:
            self._intersections[action - 16] = -1

        if self.current_player == 1:
            self._player1_walls_remaining -= 1
        else:
            self._player2_walls_remaining -= 1
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
        HORIZONTAL = 1
        VERTICAL = -1

        valid = []
        opponent_north = location == opponent_loc - 5
        opponent_south = location == opponent_loc + 5
        opponent_east = location == opponent_loc - 1
        opponent_west = location == opponent_loc + 1

        current_row = location // self.N_ROWS

        intersections = self._get_intersections(walls, location)
        n = intersections['NW'] != HORIZONTAL and intersections['NE'] != HORIZONTAL and not opponent_north
        s = intersections['SW'] != HORIZONTAL and intersections['SE'] != HORIZONTAL and not opponent_south
        e = intersections['NE'] != VERTICAL and intersections['SE'] != VERTICAL and not opponent_east
        w = intersections['NW'] != VERTICAL and intersections['SW'] != VERTICAL and not opponent_west
        if n or (player == 1 and current_row == 4): valid.append(self._DIRECTIONS['N'])
        if s or (player == 2 and current_row == 0): valid.append(self._DIRECTIONS['S'])
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
        n_border = current_tile > 19
        e_border = current_tile % 5 == 4
        s_border = current_tile < 5
        w_border = current_tile % 5 == 0

        if n_border:
            ne_intersection = 1
            if w_border:
                nw_intersection = -1
                sw_intersection = -1
                se_intersection = intersections[(current_tile - 5) - (location_row - 1)]
            elif e_border:
                nw_intersection = 1
                se_intersection = -1
                if current_tile > 24:
                    print(current_tile)
                sw_intersection = intersections[(current_tile - 5) - (location_row - 1) - 1]
            else:
                nw_intersection = 1
                if current_tile > 24:
                    print(current_tile)
                sw_intersection = intersections[(current_tile - 5) - (location_row - 1) - 1]
                se_intersection = intersections[(current_tile - 5) - (location_row - 1)]
        elif s_border:
            sw_intersection = 1
            if w_border:
                nw_intersection = -1
                se_intersection = 1
                ne_intersection = intersections[current_tile - location_row]
            elif e_border:
                se_intersection = -1
                ne_intersection = -1
                nw_intersection = intersections[current_tile - location_row - 1]
            else:
                se_intersection = 1
                ne_intersection = intersections[current_tile - location_row]
                nw_intersection = intersections[current_tile - location_row - 1]


        # West but not north or south
        elif w_border:
            nw_intersection = -1
            sw_intersection = -1
            ne_intersection = intersections[current_tile - location_row]
            se_intersection = intersections[(current_tile - 5) - (location_row - 1)]

        elif e_border:
            ne_intersection = -1
            se_intersection = -1
            nw_intersection = intersections[current_tile - location_row - 1]
            sw_intersection = intersections[(current_tile - 5) - (location_row - 1) - 1]

        # No borders
        else:
            ne_intersection = intersections[current_tile - location_row]
            nw_intersection = intersections[current_tile - location_row - 1]
            sw_intersection = intersections[(current_tile - 5) - (location_row - 1) - 1]
            se_intersection = intersections[(current_tile - 5) - (location_row - 1)]

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
                valid.append(ix + 16)

        return valid

    def _validate_horizontal(self, ix):
        column = ix % 4

        if self._intersections[ix] != 0:
            return False

        if column != 0:
            if self._intersections[ix - 1] == 1:
                return False

        if column != 3:
            if self._intersections[ix + 1] == 1:
                return False

        return not self._blocks_path(ix, self.HORIZONTAL)

    def _validate_vertical(self, ix):
        row = ix // 4
        if self._intersections[ix] != 0:
            return False

        if row != 0:
            if self._intersections[ix - 4] == -1:
                return False

        if row != 3:
            if self._intersections[ix + 4] == -1:
                return False

        return not self._blocks_path(ix, self.VERTICAL)

    def _blocks_path(self, wall_location, orientation):
        player1_target = 4
        player2_target = 0

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
        invalid_rows = [5, -1]
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
                    new_position = current_position + 5
                elif direction == self._DIRECTIONS['S']:
                    new_position = current_position - 5
                elif direction == self._DIRECTIONS['E']:
                    new_position = current_position + 1
                elif direction == self._DIRECTIONS['W']:
                    new_position = current_position - 1
                elif direction == self._DIRECTIONS['NN']:
                    new_position = current_position + 10
                elif direction == self._DIRECTIONS['SS']:
                    new_position = current_position - 10
                elif direction == self._DIRECTIONS['EE']:
                    new_position = current_position + 2
                elif direction == self._DIRECTIONS['WW']:
                    new_position = current_position - 2
                elif direction == self._DIRECTIONS['NE']:
                    new_position = current_position + 6
                elif direction == self._DIRECTIONS['NW']:
                    new_position = current_position + 4
                elif direction == self._DIRECTIONS['SW']:
                    new_position = current_position - 6
                elif direction == self._DIRECTIONS['SE']:
                    new_position = current_position - 4
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

    def add_wall(self, wall, orientation):
        self._intersections[wall] = orientation

    def print_board(self):
        player1_row = self._positions[1] // 5 * 2
        player1_col = self._positions[1] % 5 * 2
        player2_row = self._positions[2] // 5 * 2
        player2_col = self._positions[2] % 5 * 2

        x = 'X'
        o = 'O'

        v = 'v'
        h = 'h'
        dash = '-'
        none = ''

        grid_new = np.zeros([9, 9])
        grid_new[player1_row, player1_col] = 1
        grid_new[player2_row, player2_col] = 2
        
        i_reshaped = self._intersections.reshape([4, 4])
        for i in range(4):
            for j in range(4):
                if i_reshaped[i][j] == 1:
                    grid_new[((i * 2) + 1), (j * 2):((j * 2) + 3)] = 3
                elif i_reshaped[i][j] == -1:
                    grid_new[(i * 2):((i * 2) + 3), ((j * 2) + 1)] = 4

        for i in range(9):
            render_row = ""
            for j in range(9):
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
        """
        """
        self.reset()     #
        p1, p2 = self.players
        states, mcts_probs, current_players = [], [], []       # 

        while(1):   # 
            # 
            tic = time.time()
            move, move_probs = player.choose_action(self, temp=temp, return_prob=1)  #
            toc = time.time()
            print('[Move probs]\n', move_probs[:12])
            print('[Wall probs]\n', move_probs[12:])
            print("player %s  chosed move : %s ,prob: %.3f  spend: %.2f seconds" % (self.current_player, move, move_probs[move], (toc-tic)))
            # 
            states.append(self.state())
            mcts_probs.append(move_probs)
            current_players.append(self.current_player)
            # 
            self.step(move)
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
