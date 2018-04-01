
"""
Created on Thu Mar 29 10:53:43 2018

@author: tomor
"""

from __future__ import print_function
import numpy as np

def play_game(p1,p2,env,draw=False):
    current_player = None
    while not env.game_over():
        
        if current_player == p1:
            current_player = p2
        else:
            current_player = p1
            
        if draw:
           if draw == 1 and current_player == p1:
               env.draw_board()
           if draw == 2 and current_player == p2:
               env.draw_board()
               
        current_player.take_action(env)
        
        state = env.get_state()
        p1.update_state_history(state)
        p2.update_state_history(state)
        
        if draw:
            env.draw_board
        
    p1.update(env)
    p2.update(env)
    return env.winner
        
def initialV_x(env, state_winner_triples):
    V = np.zeros(env.num_states)
    for state,winner,ended in state_winner_triples:
        if ended:
            if winner == env.x:
                v = 1
            else:
                v = 0
        else:
            v = 0.5
        
        V[state] = v
    return V

def initialV_o(env,state_winner_triples):
    V = np.zeros(env.num_states)
    for state,winner,ended in state_winner_triples:
        if ended:
            if winner == env.o:
                v = 1
            else:
                v = 0
        else:
            v = 0.5
        
        V[state] = v
    return V

def get_state_hash_and_winner(env,i=0,j=0):
    results = []
    
    for v in (0,env.x,env.o):
        env.board[i,j] = v
        if j == 2:
            if i == 2:
                state = env.get_state()
                ended = env.game_over(force_recalculate=True)
                winner = env.winner
                results.append((state, winner, ended))
            else:
                results += get_state_hash_and_winner(env, i+1,0)
        else:
            results += get_state_hash_and_winner(env,i,j+1)
    
    return results
       
class Env():
    
    def __init__(self):
        self.board=np.zeros((3,3),dtype='int8')
        self.x=-1
        self.o=1
        self.winner = None
        self.ended = False
        self.num_states = 3**(9)
        self.states = np.asarray([[x,0.5] for x in range(0,19683)])   
        
    def is_empty(self,i,j):
        return self.board[i,j]==0
    
    def reward(self,sym):
        if not self.game_over():
            return 0
        
        if self.winner == sym:
            r=1
        elif self.winner == None:
            r=0.5
        else:
            r=0
        return r 
        
    def draw_board(self):
        for i in range(0,3):
            print('-------------')
            for j in range(0,3):
                print(' ',end='')
                if self.board[i,j]==self.x:
                    print('x |',end='')
                elif self.board[i,j]==self.o:
                    print('o |',end='')
                else:
                    print('  |',end='')
            print('')
        print('-------------')

    def get_state(self):
        k=0
        h=0
        for i in range(3):
            for j in range(3):
                
                if self.board[i,j]==0:
                    v=0
                elif self.board[i,j]==self.x:
                    v=1
                elif self.board[i,j]==self.o:
                    v=2
                
                h+=(3**k)*v
                k+=1
        return h
    
    def game_over(self,force_recalculate=False):
        
        
        for i in range(0,3):
            for player in (self.x,self.o):
                if self.board[i].sum() == player*3:
                    self.winner=player
                    self.ended=True
                    return True
                
        for j in range(0,3):
            for player in (self.x,self.o):
                if self.board[:,j].sum()==player*3:
                    self.winner = player
                    self.ended = True
                    return True
                
        for player in (self.x,self.o):
            if self.board.trace() == player * 3:
                self.winner=player
                self.ended=True
                return True
        
            if np.fliplr(self.board).trace() == player * 3:
                self.winner = player
                self.ended = True
                return True
            
        if np.all(self.board != 0):
            self.winner = None
            self.ended = True
            return True
            
        self.winner = None
        return False
    
class Agent():
    
    def __init__(self,eps=0.1,alpha=0.5):
        self.eps = eps
        self.alpha=alpha
        self.verbose = False
        self.state_history = []

    def setV(self,V):
        self.V = V
        
    def set_symbol(self,sym):
        self.sym = sym
        
    def set_verbose(self,v):
        self.verbose = v
        
    def reset_history(self):
        self.state_history=[]
        
    def update_state_history(self,s):
        self.state_history.append(s)
    
    def update(self,env):
        reward = env.reward(self.sym)
        target = reward
        for prev in reversed(self.state_history):
            value = self.V[prev] + self.alpha*(target - self.V[prev])
            self.V[prev] = value
            target=value
        self.reset_history()
    
    def take_action(self,env):
        r = np.random.rand()
        best_state = None
        if r < self.eps:
            if self.verbose:
                print('Taking a random action')
            
            possible_moves = []
            for i in range(0,3):
                for j in range(0,3):
                    if env.is_empty(i,j):
                        possible_moves.append((i,j))
            idx = np.random.choice(len(possible_moves))
            next_move = possible_moves[idx]
        else:
            pos2value = {}
            next_move = None
            best_value = -1
            for i in range(0,3):
                for j in range(0,3):
                    if env.is_empty(i,j):
                        env.board[i,j] = self.sym
                        state = env.get_state()
                        env.board[i,j] = 0
                        pos2value[(i,j)] = self.V[state]
                        if self.V[state] > best_value:
                            best_value = self.V[state]
                            best_state = state
                            next_move = (i,j)
            if self.verbose:
                print('Taking a greedy action')
                for i in range(0,3):
                    print('-------------')
                    for j in range(0,3):
                        if env.is_empty(i,j):
                            print('{0:.2f}|'.format(pos2value[(i,j)]),end='')
                        else:
                            print(' ',end='')
                            if env.board[i,j] == env.x:
                                print('x |',end='')
                            elif env.board[i,j] == env.o:
                                print('o |',end='')
                            else:
                                print('  |',end='')
                    print('')
                print('-------------')
        
        env.board[next_move] = self.sym
        
class Human:
    def __init__(self):
        pass
    
    def take_action(self,env):
        while True:
            move = input("Enter coordinates i,j for your next move 'i,j'; i=0..2,j=0..2:")
            i,j = move.split(',')
            i = int(i)
            j=int(j)
            
            if i<3 and j <3:
                if env.is_empty(i,j):
                    env.board[i,j] = self.sym
                    break
                else:
                    print('INVALID MOVE: non-empty position')
            else:
                print('INVALID MOVE: coordinates out of range')
                
    def update(self,env):
        pass
    
    def update_state_history(self,s):
        pass
    
    def set_symbol(self,sym):
        self.sym = sym

if __name__ == '__main__':
    p1 = Agent()
    p2 = Agent()
    
    env = Env()
    state_winner_triples = get_state_hash_and_winner(env)
    
    Vx= initialV_x(env,state_winner_triples)
    p1.setV(Vx)
    Vo = initialV_o(env,state_winner_triples)
    p2.setV(Vo)
    
    p1.set_symbol(env.x)
    p2.set_symbol(env.o)
    
    ##Train time
    T = input('How many games should we train opponent? (int) : ')
    T = int(T)
    for t in range(T):
        if t % 200 == 0:
            print(t)
        play_game(p1,p2,Env())
        
    human = Human()
    human.set_symbol(env.o)
    
    wins=0
    games=0
    lost=0
    draws=0
    while True:
        p1.set_verbose(True)
        p2.set_verbose(True)
        
        env = Env()
        if human.sym == env.o:
            winner = play_game(p1,human,env,draw=2)
        elif human.sym == env.x:
            winner = play_game(human,p2,env,draw=1)
        
        if human.sym == env.winner:
            wins+=1
        elif env.winner == None:
            draws+=1
        else:
            lost+=1
            
        games +=1
        
        record={'games':games,'wins':wins,'draws':draws,'lost':lost}
        print('Games Played: {games}  W:{wins} D:{draws} L:{lost}'.format(**record))
        answer = input('Play again? [y/n]: ')
        if answer and answer.lower()[0] == 'n':
            break
        
        if human.sym == env.o:
            human.set_symbol(env.x)
        else:
            human.set_symbol(env.o)
        
        T = input('How many more games should we train the opponent? (int) : ')
        T = int(T)
        p1.set_verbose(False)
        p2.set_verbose(False)
        for t in range(T):
            if t % 200 == 0:
                print(t)
            play_game(p1,p2,Env())
        
        None == None