# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
import logging

# python simulator.py --player_1 student_agent --player_2 random_agent --display

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
        
        
        my_step = self.alpha_beta(chess_board, my_pos, adv_pos, max_step, -sys.maxsize, sys.maxsize, 0, 1)

        time_taken = time.time() - start_time
        print("My AI's turn took ", time_taken, "seconds.")

        # dummy return
        return my_step 
    #TODO: make sure this properly works
    
    def check_endgame(self, chess_board, my_pos, adv_pos):
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

        dirs = ((-1, 0), (0, 1), (1, 0), (0, -1))
        board_size = len(chess_board)


        # Union-Find
        father = dict()
        for r in range(board_size):
            for c in range(board_size):
                father[(r, c)] = (r, c)

        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[pos1] = pos2

        for r in range(board_size):
            for c in range(board_size):
                # Check down and right
                for dir, move in enumerate(
                    dirs[1:3]
                ):  # Only check down and right
                    if chess_board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(board_size):
            for c in range(board_size):
                find((r, c))
        p0_r = find(my_pos)
        p1_r = find(adv_pos)
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
        # function to calculate number of steps away from adv_pos
        endgame = self.check_endgame(chess_board, my_pos, adv_pos)
        if endgame[0]:
            if endgame[1] > endgame[2]:
                return 20000
            elif endgame[1] == endgame[2]:
                return -10000
            else:
                return -20000
        else:
            # calculate advs possible moves
            moves = self.get_player_moves(chess_board, adv_pos, my_pos, max_step)
            cur_move_score -= len(moves)
            # check if my_pos is adjacent to two walls
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
   
    def get_player_moves(self, chess_board, my_pos, adv_pos, max_step):
        moves = []
        dirs = ((-1, 0), (0, 1), (1, 0), (0, -1))

        for s in range(max_step):
            self.take_steps(chess_board, my_pos, adv_pos, s, dirs, moves)
        return moves
    
    def take_steps(self,chess_board, my_pos, adv_pos, s, dirs, moves):
        # Get all possible moves for the current player


        r,c = my_pos
        if s != 0:
            for d in range(0,4):
                # check if the move is valid
                my_pos = (r+dirs[d][0], c+dirs[d][1])
                if not chess_board[r, c, d] and not adv_pos == my_pos:
                    self.take_steps(chess_board, my_pos, adv_pos, s-1, dirs, moves)
        else:
            my_pos_present = False
            for pos in moves:
                if pos is None:
                    moves.remove(pos)
                    continue
                if np.array_equal(my_pos, pos[0]):
                    my_pos_present = True
                    break
            if not my_pos_present:
                for d in range(0,4):
                    # check where can i place a wall
                    if not chess_board[r, c, d]:
                        moves.append((my_pos, d))
            return moves

    
    
    def alpha_beta(self, chess_board, my_pos, adv_pos, max_step, alpha, beta, depth, max_depth):
        """
        Implement the alpha-beta pruning algorithm here.
        """
        # check if the game is over
        if depth == max_depth or self.check_endgame(chess_board, my_pos, adv_pos)[0]:

            return my_pos, self.heuristic(chess_board, my_pos, adv_pos, max_step)
        # get all possible moves for the current player
        moves = self.get_player_moves(chess_board, my_pos, adv_pos, max_step)
        best_move = moves[0]
        # if it is my turn
        if depth % 2 == 0:
            for move in moves:
                # get the next state
                next_state = self.alpha_beta(chess_board, move[0], adv_pos, max_step, alpha, beta, depth+1, max_depth)
                # check if the next state is better than the current state
                if next_state[1] > alpha:
                    alpha = next_state[1]
                    best_move = move
                # check if the next state is better than the current state
                if beta <= alpha:
                    break
            return best_move, alpha
        # if it is the adversary's turn
        else:
            for move in moves:
                # get the next state
                next_state = self.alpha_beta(chess_board, my_pos, move[0], max_step, alpha, beta, depth+1, max_depth)
                # check if the next state is better than the current state
                if next_state[1] < beta:
                    beta = next_state[1]
                    best_move = move
                # check if the next state is better than the current state
                if beta <= alpha:
                    break
            return best_move,beta

