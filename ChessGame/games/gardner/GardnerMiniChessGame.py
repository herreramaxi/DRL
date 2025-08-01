from __future__ import print_function
import sys

from ChessGame.games.gardner.GardnerMiniChessLogic import Board
sys.path.append("/Users/shiningsunnyday/Desktop/2021-2022/Fall Quarter/AA 228/Final Project/mcts-chess")
from ChessGame.games.game import Game
import numpy as np
import time
import hashlib

"""
Game class implementation for the game of TicTacToe.

Author: Karthik selvakumar, github.com/karthikselva
Date: May 15, 2018.
"""

class GardnerMiniChessGame(Game):
    RECURSION_LIMIT = 1000
    def __init__(self, n=5):
        self.n = n
        self.setAllActions()
    def getInitBoard(self):
        sys.setrecursionlimit(GardnerMiniChessGame.RECURSION_LIMIT)
        # return initial board (numpy board)
        b = Board(self.n,
            [
                #  Representation of Gardner Board
                #
                # 5    ♜ ♞ ♝ ♛ ♚
                # 4    ♟ ♟ ♟ ♟ ♟
                # 3    ·  ·  ·  ·  ·
                # 2    ♙ ♙ ♙ ♙ ♙
                # 1    ♖ ♘ ♗ ♕ ♔
                #
                #      a  b  c  d  e

                [-Board.ROOK, -Board.KNIGHT, -Board.BISHOP, -Board.QUEEN, -Board.KING],
                [-Board.PAWN, -Board.PAWN, -Board.PAWN, -Board.PAWN, -Board.PAWN],
                [Board.BLANK, Board.BLANK,   Board.BLANK,   Board.BLANK, Board.BLANK],
                [Board.PAWN, Board.PAWN, Board.PAWN, Board.PAWN, Board.PAWN],
                [Board.ROOK, Board.KNIGHT, Board.BISHOP, Board.QUEEN, Board.KING],
            ]
        )
        return b.pieces_without_padding()

    def setAllActions(self):
        self.action_to_id = {}
        self.id_to_action = {}

        tmp_board = Board(self.n,[
            [Board.BLANK, Board.BLANK, Board.BLANK, Board.BLANK, Board.BLANK],
            [Board.BLANK, Board.BLANK, Board.BLANK, Board.BLANK, Board.BLANK],
            [Board.BLANK, Board.BLANK, Board.BLANK, Board.BLANK, Board.BLANK],
            [Board.BLANK, Board.BLANK, Board.BLANK, Board.BLANK, Board.BLANK],
            [Board.BLANK, Board.BLANK, Board.BLANK, Board.BLANK, Board.BLANK],
        ])
        piece_types = [Board.ROOK, Board.KNIGHT, Board.BISHOP, Board.QUEEN, Board.KING,Board.PAWN]
        # piece_types = [Board.PAWN]
        id = 0
        for i,piece in enumerate(piece_types):
            for row in range(0,self.n):
                for col in range(0,self.n):
                    tmp_board.set(row,col,piece)
                    for (p,start,end) in tmp_board._get_legal_moves(1):
                        key = str(piece)+":"+str(start)+":"+str(end)
                        self.action_to_id[key] = id
                        self.id_to_action[id] = (piece,start,end)
                        id += 1
                    tmp_board.set(row,col,Board.BLANK)

        piece = Board.PAWN
        for row in range(0,self.n):
            for col in range(0,self.n):
                tmp_board.set(row,col,piece)
                if piece == Board.PAWN:
                    if col > 0:
                        if row < (self.n-1):
                            tmp_board.set(row+1,col-1,-Board.PAWN)
                        if row > 0:
                            tmp_board.set(row-1,col-1,-Board.PAWN)
                    if col < (self.n-1):
                        if row < (self.n-1):
                            tmp_board.set(row+1,col+1,-Board.PAWN)
                        if row > 0:
                            tmp_board.set(row-1,col+1,-Board.PAWN)
                for (p,start,end) in tmp_board._get_legal_moves(1):
                    key = str(piece)+":"+str(start)+":"+str(end)
                    if self.action_to_id.get(key) == None:
                        self.action_to_id[key] = id
                        self.id_to_action[id] = (piece,start,end)
                        id += 1

    def getBoardSize(self):
        # (a,b) tuple
        return (self.n, self.n)

    def getActionSize(self):
        # return number of actions
        return len(self.action_to_id)


    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        b = Board(self.n,board)
        move = self.id_to_action[action]
        b.execute_move(move,player)
        return (b.pieces_without_padding(), -player)

    def getValidMoves(self, board, player, return_type="one_hot"):
        # return a fixed size binary vector
        valids = [0.0]*self.getActionSize()
        b = Board(self.n,board)
        if not b.has_legal_moves(player):
            valids[-1]=1.0
            return np.array(valids)
        move_list = list()
        for (p, x, y) in b.get_legal_moves(player):
            key = str(p)+":"+str(x)+":"+str(y)
            move_list.append(self.action_to_id[key])
            valids[self.action_to_id[key]] = 1.0
        if return_type == "one_hot":
            return np.array(valids, dtype=np.int32)
        else:
            return move_list

    def getGreedyMove(self,board,player):
        b = Board(self.n, board)
        p,x,y = b.greedy_move(player)
        key = str(p) + ":" + str(x) + ":" + str(y)
        return self.action_to_id[key]

    def getRandomMove(self, board, player):
        b = Board(self.n, board)
        p,x,y = b.random_move(player)
        key = str(p) + ":" + str(x) + ":" + str(y)
        return self.action_to_id[key]

    def getGameEnded(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        b = Board(self.n,board)
        if b.is_win(player):
            return 1
        if b.is_win(-player):
            return -1
        if b.has_legal_moves(player):
            return 0
        # draw has a very little value
        # return 1e-4
        return 0.5

    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        value =  [[j*player for j in i] for i in board]
        return value

    def getSymmetries(self, board, pi):
        # mirror, rotational
        # assert(len(pi) == self.n**2+1)  # 1 for pass
        return [(board, pi)]

    def stringRepresentation(self, board):
        return hashlib.md5(np.array_str(np.array(board)).encode('utf-8')).hexdigest()

    def display(self, board, player):
        return Board(self.n, board).display(player)

    def get_action_humanized(self, action_id, player):                
        action = self.id_to_action[action_id]
        if action is None:
            return "Invalid Action"
        
        piece = action[0]
        piece_humanized = self.get_piece_humamized(piece, player)
        source = self.get_position_humanized(action[1])
        target = self.get_position_humanized(action[2])
        return f"{piece_humanized} {source}-{target}"
    
    square_map = {
        15: "a5", 16: "b5", 17: "c5", 18: "d5", 19: "e5",
        22: "a4", 23: "b4", 24: "c4", 25: "d4", 26: "e4",
        29: "a3", 30: "b3", 31: "c3", 32: "d3", 33: "e3",
        36: "a2", 37: "b2", 38: "c2", 39: "d2", 40: "e2",
        43: "a1", 44: "b1", 45: "c1", 46: "d1", 47: "e1"}
    def get_position_humanized(self, position):
        return self.square_map.get(position, "Invalid Position")

    def get_piece_humamized(self, piece, player):
        if player == 1:
            if piece == Board.ROOK:
                return '♖'
            elif piece == Board.KNIGHT:
                return '♘'
            elif piece == Board.BISHOP:
                return '♗'
            elif piece == Board.QUEEN:
                return '♕'
            elif piece == Board.KING:
                return '♔'
            elif piece == Board.PAWN:
                return '♙'
            else:
                return "Unknown Piece"
        else: # player == -1
            if piece == Board.ROOK:
                return '♜'
            elif piece == Board.KNIGHT:
                return '♞'
            elif piece == Board.BISHOP:
                return '♝'
            elif piece == Board.QUEEN:
                return '♛'
            elif piece == Board.KING:
                return '♚'
            elif piece == Board.PAWN:
                return '♟'   
            else:
                return "Unknown Piece"

def display(game,board,player):
    Board(game.n,board).display(player)

if __name__ == "__main__":
    g=GardnerMiniChessGame(n=5)
    TEST_BOARD = [[-479, -280, -320, -929, -60000], [-100, -100, -100, -100, -100], [0, 0, 0, 0, 0], [100, 100, 100, 100, 100], [479, 280, 320, 929, 60000]]
    board = Board(5, TEST_BOARD)
    board.rotate(board.pieces)
    g = GardnerMiniChessGame()


