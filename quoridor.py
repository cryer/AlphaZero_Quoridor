import numpy as np
from queue import Queue
import time

class Quoridor(object):
    HORIZONTAL = 1
    VERTICAL = -1

    def __init__(self, safe=False):
        self.safe = safe

        self.action_space = 140  # 140 possible actions in total
        self.n_players = 2
        self.players = [1, 2]  # 两个玩家
        self.reset()

    # 待改
    def load(self, p1, p2):
        self.player1 = p1
        self.player2 = p2

    # 获取当前玩家
    def get_current_player(self):
        return self.current_player

    def reset(self):
        self.current_player = 1
        self.last_player = -1

        # Initialize Tiles
        self.tiles = np.zeros(81)

        # Initialize Player Locations
        self._positions = {
            1: 4,
            2: 76
        }

        self._DIRECTIONS = {
            'N': 0, 'S': 1, 'E': 2, 'W': 3,
            'NN': 4, 'SS': 5, 'EE': 6, 'WW': 7,
            'NE': 8, 'NW': 9, 'SE': 10, 'SW': 11
        }
        self.N_DIRECTIONS = 12
        self.N_TILES = 81
        self.N_ROWS = 9
        self.N_INTERSECTIONS = 64

        # There are 64 possible intersection
        # Horizontal Walls - 1
        # No Wall - 0
        # Vertical Wall - -1
        self._intersections = np.zeros(64)

        self._player1_walls_remaining = 10
        self._player2_walls_remaining = 10

    def state(self):
        """Returns a set of 9x9 planes that represent the game state.
        1. The current player position
        2. The opponent position
        3. Vertical Walls
        4. Horizontal Walls
        5 - 14. Number of walls remaining for current player
        15 - 24. Number of walls remaining for opponent
        25. Whose turn it is (0 for player 1, 1 for player 2)
        """
        player1_position_plane = self.tiles.copy()
        player1_position_plane[self._positions[1]] = 1
        player1_position_plane = player1_position_plane.reshape([9, 9])

        player2_position_plane = self.tiles.copy()
        player2_position_plane[self._positions[2]] = 1
        player2_position_plane = player2_position_plane.reshape([9, 9])

        player1_walls_plane = np.zeros([10, 9, 9])
        player2_walls_plane = np.zeros([10, 9, 9])

        player1_walls_plane[self._player1_walls_remaining - 1, :, :] = 1
        player2_walls_plane[self._player2_walls_remaining - 1, :, :] = 1

        # Set the wall planes
        vertical_walls = np.pad(
            np.int8(self._intersections == -1).reshape([8, 8]),
            (0, 1),
            mode='constant',
            constant_values=0
        )

        horizontal_walls = np.pad(
            np.int8(self._intersections == 1).reshape([8, 8]),
            (0, 1),
            mode='constant',
            constant_values=0
        )

        no_walls = np.pad(
            np.int8(self._intersections == 0).reshape([8, 8]),
            (0, 1),
            mode='constant',
            constant_values=0
        )

        # 不同玩家调整平面顺序
        if self.current_player == 1:
            state = np.stack([
                no_walls,
                vertical_walls,
                horizontal_walls,
                player1_position_plane,
                player2_position_plane,
            ])

            # print('Shape is {shape}'.format(shape=state.shape))

            current_player_plane = np.zeros([1, 9, 9])
            state = np.vstack([state, player1_walls_plane, player2_walls_plane, current_player_plane])

        if self.current_player == 2:
            state = np.stack([
                no_walls,
                vertical_walls,
                horizontal_walls,
                player2_position_plane,
                player1_position_plane,
            ])

            current_player_plane = np.ones([1, 9, 9])
            state = np.vstack([state, player2_walls_plane, player1_walls_plane, current_player_plane])
            # print(state.shape)
        return state

    def load_state(self, state):
        """Mutates the Quoridor object to match a given state"""
        current_player = state[-1] == np.zeros([9, 9])
        # TODO: Implement the rest of this

    def actions(self):
        player = self.current_player
        location = self._positions[player]

        opponent = 1 if player == 2 else 2
        opponent_loc = self._positions[opponent]
        walls = self._intersections  # 长64一维数组
        # 获得合法棋子动作空间
        pawn_actions = self._valid_pawn_actions(location=location,
                                                opponent_loc=opponent_loc, walls=walls, player=player)
        # 如果当前为玩家1并且还有挡板，或者玩家为2，并且还有挡板，则获取挡板的合法动作空间
        if ((self.current_player == 1 and self._player1_walls_remaining > 0)
            or (self.current_player == 2 and self._player2_walls_remaining > 0)):
            wall_actions = self._valid_wall_actions()  # 获得合法挡板动作空间

            # 调整+12   因为前12个是棋子动作 4+8
            wall_actions = [action + 12 for action in wall_actions]
        else:
            wall_actions = []
        return pawn_actions + wall_actions

    def step(self, action):
        """Take a step in the environment given the current action"""
        # self._logger.info("Player {player} chooses action {action}".format(player=self.current_player, action=action))
        player = self.current_player
        done = False
        # 添加
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
            print("game over !winner is player" + str(winner))
            done = True
        else:
            self.rotate_players()  # 切换玩家
            # observation = self.get_state  # get_state未实现
            # observation = self.state()
            # print("game over !winner is player" + str(winner))

        return done

    # 判断游戏是否结束
    def game_end(self):
        pass

    # 判断是否有胜者
    def has_a_winner(self):
        game_over = False
        winner = None
        if self._positions[2] < 9:
            winner = 2
            game_over = True
        elif self._positions[1] > 71:
            winner = 1
            game_over = True
        return game_over, winner

    # 获取奖励
    def _get_rewards(self):
        done = True
        if self._positions[2] < 9:
            rewards, done = (1, -1)
        elif self._positions[1] > 71:
            rewards = (-1, 1)
        else:
            rewards = (0, 0)
            done = False
        return rewards, done

    # 处理棋子动作
    def _handle_pawn_action(self, action, player):
        if action == self._DIRECTIONS['N']:
            self._positions[player] += 9
        elif action == self._DIRECTIONS['S']:
            self._positions[player] -= 9
        elif action == self._DIRECTIONS['E']:
            self._positions[player] += 1
        elif action == self._DIRECTIONS['W']:
            self._positions[player] -= 1
        elif action == self._DIRECTIONS['NN']:
            self._positions[player] += 18
        elif action == self._DIRECTIONS['SS']:
            self._positions[player] -= 18
        elif action == self._DIRECTIONS['EE']:
            self._positions[player] += 2
        elif action == self._DIRECTIONS['WW']:
            self._positions[player] -= 2
        elif action == self._DIRECTIONS['NW']:
            self._positions[player] += 8
        elif action == self._DIRECTIONS['NE']:
            self._positions[player] += 10
        elif action == self._DIRECTIONS['SW']:
            self._positions[player] -= 10
        elif action == self._DIRECTIONS['SE']:
            self._positions[player] -= 8
        else:
            raise ValueError("Invalid Pawn Action: {action}".format(action=action))

    # 处理挡板动作
    def _handle_wall_action(self, action):
        # Action values less than 64 are horizontal walls
        if action < 64:
            self._intersections[action] = 1
        # Action values above 64 are vertical walls
        else:
            self._intersections[action - 64] = -1

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

    # walls ：长64一维数组 location：int 0-80
    def _valid_pawn_actions(self, walls, location, opponent_loc, player=1):
        HORIZONTAL = 1
        VERTICAL = -1

        valid = []
        # 判断对面棋子是否相邻
        opponent_north = location == opponent_loc - 9
        opponent_south = location == opponent_loc + 9
        opponent_east = location == opponent_loc - 1
        opponent_west = location == opponent_loc + 1

        current_row = location // self.N_ROWS

        intersections = self._get_intersections(walls, location)
        # 判断北面没有水平挡板和对面棋子
        n = intersections['NW'] != HORIZONTAL and intersections['NE'] != HORIZONTAL and not opponent_north
        # 判断南面没有水平挡板和对面棋子
        s = intersections['SW'] != HORIZONTAL and intersections['SE'] != HORIZONTAL and not opponent_south
        # 判断东面没有竖直挡板和对面棋子
        e = intersections['NE'] != VERTICAL and intersections['SE'] != VERTICAL and not opponent_east
        # 判断西面没有竖直挡板和对面棋子
        w = intersections['NW'] != VERTICAL and intersections['SW'] != VERTICAL and not opponent_west
        # 向北走，两种情况：1，按照上面的判断可走 2，虽到边界但是再走可以获胜
        if n or (player == 1 and current_row == 8): valid.append(self._DIRECTIONS['N'])
        # 同理
        if s or (player == 2 and current_row == 0): valid.append(self._DIRECTIONS['S'])
        if e: valid.append(self._DIRECTIONS['E'])
        if w: valid.append(self._DIRECTIONS['W'])
        # 如果北面有对手棋子并且北面没有水平挡板
        if opponent_north and intersections['NE'] != HORIZONTAL and intersections['NW'] != HORIZONTAL:
            # 获取对手周围的挡板信息
            n_intersections = self._get_intersections(walls, opponent_loc)
            # 如果对手北面没有水平挡板，或者 玩家1在第八行，也就是倒数第二行
            if n_intersections['NW'] != HORIZONTAL and n_intersections['NE'] != HORIZONTAL \
                    or (current_row == 7 and player == 1):
                # 可以走向北两步，也就是NN
                valid.append(self._DIRECTIONS['NN'])
            # 如果对手东-北面没有竖直挡板，并且自己东-北面没有竖直挡板，可以走两步NE
            if n_intersections['NE'] != VERTICAL and intersections['NE'] != VERTICAL:
                valid.append(self._DIRECTIONS['NE'])

            if n_intersections['NW'] != VERTICAL and intersections['NW'] != VERTICAL:
                valid.append(self._DIRECTIONS['NW'])


        elif opponent_south and intersections['SE'] != HORIZONTAL and intersections['SW'] != HORIZONTAL:
            s_intersections = self._get_intersections(walls, opponent_loc)
            if s_intersections['SW'] != HORIZONTAL and s_intersections['SE'] != HORIZONTAL \
                    or (current_row == 1 and player == 2):
                valid.append(self._DIRECTIONS['SS'])

            if s_intersections['SE'] != VERTICAL and intersections['SE'] != VERTICAL:
                valid.append(self._DIRECTIONS['SE'])

            if s_intersections['SW'] != VERTICAL and intersections['SW'] != VERTICAL:
                valid.append(self._DIRECTIONS['SW'])


        elif opponent_east and intersections['SE'] != VERTICAL and intersections['NE'] != VERTICAL:
            e_intersections = self._get_intersections(walls, opponent_loc)
            if e_intersections['SE'] != VERTICAL and e_intersections['NE'] != VERTICAL:
                valid.append(self._DIRECTIONS['EE'])

            if e_intersections['NE'] != HORIZONTAL:
                valid.append(self._DIRECTIONS['NE'])

            if e_intersections['SE'] != HORIZONTAL:
                valid.append(self._DIRECTIONS['SE'])


        elif opponent_west and intersections['SW'] != VERTICAL and intersections['NW'] != VERTICAL:
            w_intersections = self._get_intersections(walls, opponent_loc)
            if w_intersections['NW'] != VERTICAL and w_intersections['SW'] != VERTICAL:
                valid.append(self._DIRECTIONS['WW'])

            if w_intersections['NW'] != HORIZONTAL:
                valid.append(self._DIRECTIONS['NW'])

            if w_intersections['SW'] != HORIZONTAL:
                valid.append(self._DIRECTIONS['SW'])

        return valid

    # intersections： 一维数组 长64   current_tile：当前位置，int 0-80
    def _get_intersections(self, intersections, current_tile):
        """Gets the four intersections for a given tile."""
        location_row = current_tile // self.N_ROWS
        # 判断棋子是否在四周边界
        n_border = current_tile > 71
        e_border = current_tile % 9 == 8
        s_border = current_tile < 9
        w_border = current_tile % 9 == 0

        if n_border:
            ne_intersection = 1
            if w_border:
                nw_intersection = -1
                sw_intersection = -1
                se_intersection = intersections[(current_tile - 9) - (location_row - 1)]
            elif e_border:
                nw_intersection = 1
                se_intersection = -1
                sw_intersection = intersections[(current_tile - 9) - (location_row - 1) - 1]
            else:
                nw_intersection = 1
                sw_intersection = intersections[(current_tile - 9) - (location_row - 1) - 1]
                se_intersection = intersections[(current_tile - 9) - (location_row - 1)]
        elif s_border:
            sw_intersection = 1
            if w_border:
                nw_intersection = -1
                se_intersection = 1
                ne_intersection = intersections[current_tile - location_row]
            elif e_border:
                se_intersection = -1
                ne_intersection = -1
                nw_intersection = ne_intersection = intersections[current_tile - location_row - 1]
            else:
                se_intersection = 1
                ne_intersection = intersections[current_tile - location_row]
                nw_intersection = ne_intersection = intersections[current_tile - location_row - 1]


        # West but not north or south
        elif w_border:
            nw_intersection = -1
            sw_intersection = -1
            ne_intersection = intersections[current_tile - location_row]
            se_intersection = intersections[(current_tile - 9) - (location_row - 1)]

        elif e_border:
            ne_intersection = -1
            se_intersection = -1
            nw_intersection = intersections[current_tile - location_row - 1]
            sw_intersection = intersections[(current_tile - 9) - (location_row - 1) - 1]

        # No borders
        else:
            ne_intersection = intersections[current_tile - location_row]
            nw_intersection = intersections[current_tile - location_row - 1]
            sw_intersection = intersections[(current_tile - 9) - (location_row - 1) - 1]
            se_intersection = intersections[(current_tile - 9) - (location_row - 1)]

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
                valid.append(ix + 64)

        return valid

    def _validate_horizontal(self, ix):
        column = ix % 8

        if self._intersections[ix] != 0:
            return False

        if column != 0:
            if self._intersections[ix - 1] == 1:
                return False

        if column != 7:
            if self._intersections[ix + 1] == 1:
                return False

        return not self._blocks_path(ix, self.HORIZONTAL)

    def _validate_vertical(self, ix):
        row = ix // 8
        if self._intersections[ix] != 0:
            return False

        if row != 0:
            if self._intersections[ix - 8] == -1:
                return False

        if row != 7:
            if self._intersections[ix + 8] == -1:
                return False

        return not self._blocks_path(ix, self.VERTICAL)

    def _blocks_path(self, wall_location, orientation):
        player1_target = 8
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
        invalid_rows = [9, -1]
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
                    new_position = current_position + 9
                elif direction == self._DIRECTIONS['S']:
                    new_position = current_position - 9
                elif direction == self._DIRECTIONS['E']:
                    new_position = current_position + 1
                elif direction == self._DIRECTIONS['W']:
                    new_position = current_position - 1
                elif direction == self._DIRECTIONS['NN']:
                    new_position = current_position + 18
                elif direction == self._DIRECTIONS['SS']:
                    new_position = current_position - 18
                elif direction == self._DIRECTIONS['EE']:
                    new_position = current_position + 2
                elif direction == self._DIRECTIONS['WW']:
                    new_position = current_position - 2
                elif direction == self._DIRECTIONS['NE']:
                    new_position = current_position + 10
                elif direction == self._DIRECTIONS['NW']:
                    new_position = current_position + 8
                elif direction == self._DIRECTIONS['SW']:
                    new_position = current_position - 10
                elif direction == self._DIRECTIONS['SE']:
                    new_position = current_position - 8
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
        player1_row = self._positions[1] // 9
        player1_col = self._positions[1] % 9
        player2_row = self._positions[2] // 9
        player2_col = self._positions[2] % 9

        x = 'X'
        o = 'O'

        v = 'v'
        h = 'h'
        dash = '-'
        none = ''

        grid = [['{dash:4}'.format(dash=dash) for i in range(9)] for i in range(9)]
        i_reshaped = self._intersections.reshape([8, 8])

        grid[player1_row][player1_col] = '{x:4}'.format(x=x)
        grid[player2_row][player2_col] = '{o:4}'.format(o=o)

        intersection_row = 7
        for i in range(8, -1, -1):
            for j in range(9):
                print(grid[i][j], end='')
            print()
            if intersection_row >= 0:
                print('{none:2}'.format(none=none), end='')
                for j in i_reshaped[intersection_row, :]:
                    if j == 1:
                        print('{h:4}'.format(h=h), end='')
                    elif j == -1:
                        print('{v:4}'.format(v=v), end='')
                    else:
                        print('{none:4}'.format(none=none), end='')
                intersection_row -= 1
                print()

    def clone(self):
        return Quoridor()
    # 自博弈一次
    def start_self_play(self, player, is_shown=0, temp=1e-3):
        """
         开始自博弈，也就是蒙特卡洛树搜索和蒙特卡洛树搜索之间的对抗。
         主要是为了产生数据集，训练神经网络，保存的数据形式：(state, mcts_probs, z)
        """
        self.reset()     # 初始化棋盘
        p1, p2 = self.players
        states, mcts_probs, current_players = [], [], []       # 初始化需要保存的信息，胜负情况要在模拟结束时保存

        while(1):   # 循环进行自博弈
            # 待修改
            tic = time.time()
            move, move_probs = player.choose_action(self, temp=temp, return_prob=1)  # 获取落子以及概率
            toc = time.time()
            print("player %s  chosed move : %s ,prob: %.3f  spend: %.2f seconds" % (self.current_player, move, move_probs[move], (toc-tic)))
            # 保存数据
            states.append(self.state())
            mcts_probs.append(move_probs)
            current_players.append(self.current_player)
            # 进行落子
            self.step(move)
            # if is_shown:
            #     self.graphic(self.board, p1, p2)
            end, winner = self.has_a_winner()
            if end:
                # 判断游戏是否结束 ，始终以当前玩家视角保存数据
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0        # 当前玩家的所有落子的z都设为1
                    winners_z[np.array(current_players) != winner] = -1.0       # 对手玩家的所有落子的z都设为-1
                # 重置MCTS节点
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)
