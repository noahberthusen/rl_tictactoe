import numpy as np
import random
from collections import defaultdict
import json

#Basic tic tac toe class
class ttt:
    def __init__(self):
        self.board = np.array([None] * 9)

    @staticmethod
    def check_for_win(board):
        WIN_STATES = [
            (0,1,2),
            (3,4,5),
            (6,7,8),
            (0,3,6),
            (1,4,7),
            (2,5,8),
            (0,4,8),
            (2,4,6)
        ]

        for a, b, c in WIN_STATES:
            if board[a] == board[b] == board[c] and board[a] == 1:
                return 100
            elif board[a] == board[b] == board[c] and board[a] == 2:
                return -100
        if len(ttt.legal_moves(board)) == 0:
            return 50
        else:
            return 0
    
    @staticmethod
    def legal_moves(board):
        return np.asarray(np.where(board == None)).flatten()

    def update_board(self, player, move):
        self.board[move] = player
        return self.check_for_win(self.board)      

    @staticmethod
    def display_board(board):
        def convert_board(board):
            readable_board = [' '] * 9
            for i in range(len(board)):
                if board[i] == 1:
                    readable_board[i] = 'X'
                elif board[i] == 2:
                    readable_board[i] = 'O'
            return readable_board
        board = convert_board(board)
        print(' {:1} | {:1} | {:1}'.format(board[0],board[1],board[2]))
        print('-----------')
        print(' {:1} | {:1} | {:1}'.format(board[3],board[4],board[5]))
        print('-----------')
        print(' {:1} | {:1} | {:1}'.format(board[6],board[7],board[8]))

#Random play
def random_play():
    q = defaultdict(lambda: [0,0,0,0,0,0,0,0,0])
    rewards = np.array([])
    games = [ttt() for i in range(150000)]
    epsilon = 1

    def p1_turn(game):
        def new_q(q, moves):
            new_q = [0] * 9
            for i in range(len(q)):
                if i not in moves:
                    new_q[i] = -200
                else:
                    new_q[i] = q[i]
            return new_q

        legal_moves = ttt.legal_moves(game.board)
        # epsilon greedy exploration
        if (random.uniform(0, 1) > epsilon):
            move = np.argmax(new_q(q[str(game.board)], legal_moves))
        else:
            move = np.random.choice(legal_moves)
        old_board = game.board.copy()
        
        # generating new state after player 1 and 2 move
        game.update_board(1, move)
        p2_turn(game)

        if not(np.array_equal(old_board, game.board)):
            updateQ(old_board, move, game.board, ttt.check_for_win(game.board))
    
    def p2_turn(game):
        legal_moves = ttt.legal_moves(game.board)
        if (len(legal_moves) != 0):
            move = np.random.choice(legal_moves)
            game.update_board(2, move)

    # Q(s)[a] :=r+γ maxa′Q(s′)[a′]
    # r = reward of new state i.e. game.check_for_win(s')
    # maxa'Q(s')[a'] is just querying the table for s' and picking the largest value in the returned array
    # each update just changes one value out of [0,0,0,0,0,0,0,0,0] array
    # new state needs to be after opponent moves
    def updateQ(s, a, new_s, r):
        # [0,0,0,0,0,0,0,0,0]
        q_value = q[str(s)].copy()
        q_value[a] = (0.9 * q_value[a]) + (0.1 * (r + (0.9 * max(q[str(new_s)]))))
        q[str(s)] = q_value

    for game in games:
        while (ttt.check_for_win(game.board) == 0):
            p1_turn(game)
    
        rewards = np.append(rewards, ttt.check_for_win(game.board))
        if (epsilon > 0.05):
            epsilon *= 0.999997

    fo = open("firstplayer.json", "w")
    json.dump(q, fo)
    fo.close()
    #len(q) ~ 3217 states
    print("Player 1 Wins: ", np.asarray(np.where(rewards == 100)).flatten().size/len(rewards)) # ~0.84, 0.66
    print("Player 2 Wins: ", np.asarray(np.where(rewards == -100)).flatten().size/len(rewards)) # ~0.089, 0.23
    print("Ties: ", np.asarray(np.where(rewards == 50)).flatten().size/len(rewards)) # ~0.066, 0.1
    return q

def q_play(q):
    rewards = np.array([])
    games = [ttt() for i in range(10000)]

    def p_turn(player, game):
        def new_q(q, moves):
            new_q = [0] * 9
            for i in range(len(q)):
                if i not in moves:
                    new_q[i] = -200
                else:
                    new_q[i] = q[i]
            return new_q

        legal_moves = ttt.legal_moves(game.board)
        old_board = game.board.copy()        
        if (player == 1):
            move = np.argmax(new_q(q[str(game.board)], legal_moves))
        else:
            move = np.random.choice(legal_moves)
        game.update_board(player, move)

    for game in games:
        while (ttt.check_for_win(game.board) == 0):
            p_turn(1, game)
            if (ttt.check_for_win(game.board) != 0 or len(ttt.legal_moves(game.board)) == 0):
                break
            p_turn(2, game)
    
        rewards = np.append(rewards, ttt.check_for_win(game.board))

    print()
    print("Player 1 Wins: ", np.asarray(np.where(rewards == 100)).flatten().size/len(rewards)) # ~0.977, 0.9884
    print("Player 2 Wins: ", np.asarray(np.where(rewards == -100)).flatten().size/len(rewards)) # ~0.0
    print("Ties: ", np.asarray(np.where(rewards == 50)).flatten().size/len(rewards)) # ~0.022, 0.0116

q_play(random_play())

def human_play():
    with open('aifirst.json') as f:
        q = json.load(f)
    game = ttt()

    def p_turn(game):
        def new_q(q, moves):
            new_q = [0] * 9
            for i in range(len(q)):
                if i not in moves:
                    new_q[i] = -200
                else:
                    new_q[i] = q[i]
            return new_q

        legal_moves = ttt.legal_moves(game.board)
        old_board = game.board.copy() 
        move = np.argmax(new_q(q[str(game.board)], legal_moves))
        game.update_board(1, move)

    def h_turn(game):
        ttt.display_board(game.board)
        move = input('Where would you like to play (0-8): ')
        game.update_board(2, int(move))

    while (ttt.check_for_win(game.board) == 0):
        p_turn(game)
        if (ttt.check_for_win(game.board) != 0):
            ttt.display_board(game.board)
            break
        h_turn(game)

    print()
    if (ttt.check_for_win(game.board) == 100):
        print('AI wins')
    elif (ttt.check_for_win(game.board) == -100):
        print('You somehow beat the AI')
    else:
        print('You both played a perfect game. Tie')

# human_play()