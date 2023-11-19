# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
import logging


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """

        # Some simple code to help you with timing. Consider checking 
        # time_taken during your search and breaking with the best answer
        # so far when it nears 2 seconds.
        start_time = time.time()
        time_taken = time.time() - start_time
        
        
        
        
        print("My AI's turn took ", time_taken, "seconds.")

        # dummy return
        return my_pos, self.dir_map["u"]
    #TODO: make sure this properly works
    
    def check_endgame(self):
        """
        Check if the game ends and compute the current score of the agents.

        Returns
        -------
        is_endgame : bool
            Whether the game ends.
        player_1_score : int
            The score of player 1.
        player_2_score : int
            The score of player 2.
        """
        # Union-Find
        father = dict()
        for r in range(self.board_size):
            for c in range(self.board_size):
                father[(r, c)] = (r, c)
        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[pos1] = pos2

        for r in range(self.board_size):
            for c in range(self.board_size):
                for dir, move in enumerate(
                    self.moves[1:3]
                ):  # Only check down and right
                    if self.chess_board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(self.board_size):
            for c in range(self.board_size):
                find((r, c))
        p0_r = find(tuple(self.p0_pos))
        p1_r = find(tuple(self.p1_pos))
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)
        if p0_r == p1_r:
            return False, p0_score, p1_score
        
        return True, p0_score, p1_score
    
    def get_current_player(self):
        """
        Get the positions of the current player

        Returns
        -------
        tuple of (current_player_obj, current_player_pos, adversary_player_pos)
        """
        if not self.turn:
            return self.p0, self.p0_pos, self.p1_pos
        else:
            return self.p1, self.p1_pos, self.p0_pos

    def heuristic(self,chess_board,my_pos,adv_pos,max_step):
        """
        Implement the heuristic function of your agent here.      
        """
        cur_move_score = 0
        #function to calculate number of steps away from adv_pos
        if self.check_endgame(self)[0]:
            if self.check_endgame(self)[1] > self.check_endgame(self)[2]:
                return 2000
            else:
                return -2000
        else:
            moves = self.get_player_moves(chess_board, my_pos, adv_pos, max_step)
            cur_move_score -= len(moves)
            #check if my_pos is adjacent to two walls
            if self.check_two_walls(chess_board,my_pos):
                cur_move_score -= 500
            
        return cur_move_score
    def check_two_walls(self,chess_board,my_pos):
        """
        Check if my_pos is adjacent to two walls
        """
        r,c = my_pos
        for d in range(0,4):
            for d2 in range(0,4):
                if d != d2:
                    if chess_board[r,c,d] and chess_board[r,c,d2]:
                        return True
        return False
    #TODO: change minPlayerMoves to get_player_moves
    
    def min_player_moves(self, chess_board, my_pos, adv_pos, max_step):
        """
        Get all possible moves for the current player
        """
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        steps = max_step
        # Pick steps random but allowable moves
        for _ in range(steps):
            r, c = adv_pos

            # Build a list of the moves we can make
            allowed_moves = [d
                            for d in range(0, 4)  # 4 moves possible
                            if not chess_board[r, c, d] and  # chess_board True means wall
                            not my_pos == (r + moves[d][0], c + moves[d][1])]
            # cannot move through ME
            return allowed_moves
        
    
    def alpha_beta(self, chess_board, my_pos, adv_pos, max_step, alpha, beta, depth, max_depth):
        """
        Implement the alpha-beta pruning algorithm here.
        """
        if depth == max_depth:
            return self.heuristic(chess_board, my_pos, adv_pos, max_step)
        else:
            if depth % 2 == 0:
                #max player
                moves = self.get_player_moves(chess_board, my_pos, adv_pos, max_step)
                for move in moves:
                    new_chess_board = deepcopy(chess_board)
                    new_chess_board[my_pos[0], my_pos[1], move] = True
                    new_my_pos = (my_pos[0] + self.moves[move][0], my_pos[1] + self.moves[move][1])
                    alpha = max(alpha, self.alpha_beta(new_chess_board, new_my_pos, adv_pos, max_step, alpha, beta, depth + 1, max_depth))
                    if alpha >= beta:
                        break
                return alpha
            else:
                #min player
                moves = self.min_player_moves(chess_board, my_pos, adv_pos, max_step)
                for move in moves:
                    new_chess_board = deepcopy(chess_board)
                    new_chess_board[adv_pos[0], adv_pos[1], move] = True
                    new_adv_pos = (adv_pos[0] + self.moves[move][0], adv_pos[1] + self.moves[move][1])
                    beta = min(beta, self.alpha_beta(new_chess_board, my_pos, new_adv_pos, max_step, alpha, beta, depth + 1, max_depth))
                    if alpha >= beta:
                        break
                return beta
        