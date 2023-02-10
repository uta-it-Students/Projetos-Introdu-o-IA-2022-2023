# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util
import math

from game import Agent
from pacman import GameState

from collections import namedtuple

State = namedtuple("State","score action")

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        legal_moves = gameState.getLegalActions()
        scores = [self.evaluationFunction(gameState, action) for action in legal_moves]

        bestScore = max(scores)
        max_indexes = [index for index, score in enumerate(scores) if score == bestScore]

        return legal_moves[random.choice(max_indexes)]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.
py
        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successor_game_state = currentGameState.generatePacmanSuccessor(action)
        position = successor_game_state.getPacmanPosition()
        food_list = successor_game_state.getFood().asList()
        ghost_states = successor_game_state.getGhostStates()
        ghost_positions = successor_game_state.getGhostPositions()
        scared_times = [ghostState.scaredTimer for ghostState in ghost_states]
       
        """calculates the distance to the closest food"""
    
        distances = [util.manhattanDistance(position, food) for food in food_list] 
        min_food_distance = min(distances) if distances else -1

        """Calculating the distances from packman to the ghosts"""
        ghosts_distance = 1
        ghost_proximity = 0

        for ghost_position in ghost_positions:
            distance = util.manhattanDistance(position, ghost_position)
            ghosts_distance += distance
            if distance <= 1:
                ghost_proximity += 1

        """Combining al the metrics"""
        return successor_game_state.getScore() + (1 / min_food_distance) - ghost_proximity*10  - (1 / ghosts_distance) 

        

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """

        return self.minimax(gameState, self.depth, self.index, True).action

    def random_action(self, scores, state_score, actions):
        indexes = [index for index, score in enumerate(scores) if score == state_score]
        return actions[random.choice(indexes)]

    def minimax(self, game_state: GameState, depth: int, agent_index: int, maximizing: bool) -> State:
        
        if depth == 0 or game_state.isWin() or game_state.isLose():
            return State(self.evaluationFunction(game_state), Directions.STOP)
        
        actions = game_state.getLegalActions(agent_index)
        scores = []
        last_ghost_index = game_state.getNumAgents() - 1
        is_last_ghost = agent_index == last_ghost_index 

        if is_last_ghost: 
            new_depth = depth - 1
            new_agent = 0 #return to PacMan
        else:
            new_depth = depth
            new_agent = agent_index + 1
       
        if maximizing:  
            for action in actions:
                successor_game_state = game_state.generateSuccessor(agent_index, action)
                new_state = self.minimax(successor_game_state, depth, new_agent, maximizing=False)
                scores.append(new_state.score)
        else:
            for action in actions:
                successor_game_state = game_state.generateSuccessor(agent_index, action)
                new_state = self.minimax(successor_game_state, new_depth, new_agent, maximizing=is_last_ghost)
                scores.append(new_state.score)

        selected_score = max(scores) if maximizing else min(scores)
        chosen_action = self.random_action(scores, selected_score, actions)
        return State(selected_score, chosen_action)



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def random_action(self, scores, state_score, actions):
        indexes = [index for index, score in enumerate(scores) if score == state_score]
        return actions[random.choice(indexes)]

    def minimax(self, game_state: GameState, depth: int, agent_index: int, alpha=-math.inf, beta=math.inf, maximizing = True)->State:
        if depth == 0 or game_state.isWin() or game_state.isLose():
            return State(self.evaluationFunction(game_state), Directions.STOP)
 
        actions = game_state.getLegalActions(agent_index)
        scores = []
        last_ghost_index = game_state.getNumAgents() - 1
        is_last_ghost = agent_index == last_ghost_index 

        if is_last_ghost: 
            new_depth = depth - 1
            new_agent = 0 #return to PacMan
        else:
            new_depth = depth
            new_agent = agent_index + 1

        if maximizing:
            for action in actions:
                successor_game_state = game_state.generateSuccessor(agent_index, action)
                new_state = self.minimax(successor_game_state, new_depth, new_agent, alpha, beta, False)
                scores.append(new_state.score)

                max_score = max(scores)
                alpha = max(alpha, max_score)

                if alpha > beta:
                    break
        else:
            for action in actions:
                successor_game_state = game_state.generateSuccessor(agent_index, action)
                new_state = self.minimax(successor_game_state, new_depth, new_agent, alpha, beta, is_last_ghost)
                scores.append(new_state.score)

                min_score = min(scores)
                beta = min(beta, min_score)

                if beta < alpha:
                    break
            
            
        selected_score = max(scores) if maximizing else min(scores)
        chosen_action = self.random_action(scores, selected_score, actions)
        return State(selected_score, chosen_action)


    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        return self.minimax(gameState, self.depth,0).action
        
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def random_action(self, scores, state_score, actions) -> Directions:
        indexes = [index for index, score in enumerate(scores) if score == state_score]
        return actions[random.choice(indexes)]

    def expectimax(self,game_state: GameState, depth: int, agent_index :int, maximizing: bool) -> State:
       
        if depth == 0 or game_state.isWin() or game_state.isLose():
            return State(self.evaluationFunction(game_state), Directions.STOP)

        actions = game_state.getLegalActions(agent_index)
        last_ghost_index = game_state.getNumAgents() - 1
        is_last_ghost = agent_index == last_ghost_index
       
        probability = 1 / len(actions)

        if is_last_ghost: 
            new_depth = depth - 1
            new_agent = 0 #return to PacMan
        else:
            new_depth = depth
            new_agent = agent_index + 1

        if maximizing:
            scores = []
            for action in actions:
                successor_game_state = game_state.generateSuccessor(agent_index, action)
                new_state = self.expectimax(successor_game_state, new_depth, new_agent, maximizing=False)
                scores.append(new_state.score)

            selected_score = max(scores)
            chosen_action = self.random_action(scores, selected_score, actions)
            return State(selected_score, chosen_action)
        else:
            state_values = []
            for action in actions:
                successor_game_state = game_state.generateSuccessor(agent_index, action)
                new_state = self.expectimax(successor_game_state, new_depth, new_agent, maximizing=is_last_ghost)
                state_values.append(new_state.score * probability)  
                
            return State(sum(state_values), Directions.STOP)

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        return self.expectimax(gameState, self.depth, self.index, True).action
        
        

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    """capsules = currentGameState.getCapsules()   
    newFoodList = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    pacman_pos = currentGameState.getPacmanPosition()
    current_score = currentGameState.getScore()
    ghost_score = 0
    cap_score = 0

    #Calculate the distance between pacman and capsules in game using Manhattan distance.
    #Check if capsule list is not empty.
    if(len(capsules) != 0):
        #Use manhattan distance formula
        for capsule in capsules:
            cap_dis = min([manhattanDistance(capsule, pacman_pos)])
        if cap_dis == 0 :
            cap_score = float(1)/cap_dis
        else:
            cap_score = -100
            
    #Calculate distance between ghosts and pacman using Manhattan distance.s        
    for ghost in ghostStates:
        ghost_x = (ghost.getPosition()[0])
        ghost_y = (ghost.getPosition()[1])
        ghost_pos = ghost_x,ghost_y
        ghost_dis = manhattanDistance(pacman_pos, ghost_pos)

    #Evaluation function returms following scores.
    return current_score  - (1.0/1+ghost_dis)  + (1.0/1+cap_score)"""

    capsules = currentGameState.getCapsules()   
    ghostStates = currentGameState.getGhostStates()
    pacman_pos = currentGameState.getPacmanPosition()
    current_score = currentGameState.getScore()

    distances = [manhattanDistance(capsule, pacman_pos) for capsule in capsules] 
    cap_dis = min(distances) if distances else -100
    cap_score =  1 /  1 + cap_dis
    
    ghost_distances = [manhattanDistance(pacman_pos, ghost.getPosition()) for ghost in ghostStates]
    ghost_dis = min(ghost_distances)
    

    return current_score  - (1 /1 + ghost_dis)  + (1/cap_score)
 
# Abbreviation
better = betterEvaluationFunction
