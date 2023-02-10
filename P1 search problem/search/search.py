# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from collections import namedtuple



# import cProfile
# import pstats

# def do_cprofile(func):
#     def profiled_func(*args, **kwargs):
#         profile = cProfile.Profile()
#         try:
#             profile.enable()
#             result = func(*args, **kwargs)
#             profile.disable()
#             return result
#         finally:
#             status = pstats.Stats(profile)
#             status.sort_stats(pstats.SortKey.TIME)
#             status.print_stats()
#     return profiled_func

Node = namedtuple("Node","position path") 


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        #util.raiseNotDefined()

    def isGoalState(self, state):
        """
        state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        #util.raiseNotDefined()

    def getSuccessors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        #util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        #util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem: SearchProblem)-> list:
    """
    Search the deepest nodes in the search tree first.
    
    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    #setup
    node_stack = util.Stack()
    visit_position = set()

    start_position = problem.getStartState()
    start_path = []
    start_node = Node(start_position, start_path)

    node_stack.push(start_node)

    while not node_stack.isEmpty():
        position, path = node_stack.pop()

        if problem.isGoalState(position):
            return path

        if position not in visit_position:
            visit_position.add(position)

            for new_position, direction, stepCost in problem.getSuccessors(position):
                if  new_position not in visit_position:
                    new_path =  [*path, direction]
                    new_node = Node(new_position, new_path)
                    node_stack.push(new_node)
            
            # reverse
            # neighbors = problem.getSuccessors(position)
            # neighbors.reverse()
            # for new_position, direction, stepCost in neighbors:
            #     if  new_position not in visit_position:
            #         new_path =  [*path, direction]
            #         new_node = Node(new_position, new_path)
            #         node_stack.push(new_node)

def breadthFirstSearch(problem: SearchProblem)-> list:
    """Search the shallowest nodes in the search tree first."""
    #setup
    node_queue = util.Queue()
    visit_position = set()

    start_position = problem.getStartState()
    start_path = []
    start_node = Node(start_position, start_path)

    node_queue.push(start_node)

    while not node_queue.isEmpty():
        position, path = node_queue.pop()

        if problem.isGoalState(position):
            return path

       
        if position not in visit_position:
            visit_position.add(position)

            for new_position, direction, stepCost in problem.getSuccessors(position):
                if  new_position not in visit_position:
                    new_path = [*path, direction]
                    new_node = Node(new_position, new_path)
                    node_queue.push(new_node)
        


def uniformCostSearch(problem: SearchProblem)-> list:
    """Search the node of least total cost first."""
    node_priority_queue = util.PriorityQueue()
    visit_position = set()

    start_position = problem.getStartState()
    start_path = []
    start_rout = Node(start_position, start_path)

    node_priority_queue.push(start_rout,0)

    while not node_priority_queue.isEmpty():
        position, path = node_priority_queue.pop()

        if problem.isGoalState(position):
            return path

        if position not in visit_position:
            visit_position.add(position)

            for new_position, direction, stepCost in problem.getSuccessors(position):
                if  new_position not in visit_position:
                    new_path = [*path, direction]
                    new_node = Node(new_position, new_path)
                    cost = problem.getCostOfActions(new_path)
                    node_priority_queue.push(new_node, cost)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic)-> list:
    """Search the node that has the lowest combined cost and heuristic first."""
    node_priority_queue = util.PriorityQueue()
    visit_position = set()

    start_position = problem.getStartState()
    start_path = []
    start_rout = Node(start_position, start_path)
    f_cost = heuristic(start_position, problem)

    node_priority_queue.push(start_rout, f_cost)

    while not node_priority_queue.isEmpty():
        position, path = node_priority_queue.pop()

        if problem.isGoalState(position):
            return path

        if position not in visit_position:
            visit_position.add(position)

            for new_position, direction, stepCost in problem.getSuccessors(position):
                if  new_position not in visit_position:
                    new_path = [*path, direction]
                    new_node = Node(new_position, new_path)

                    g_cost = problem.getCostOfActions(new_path)
                    h_cost = heuristic(new_position, problem)
                    f_cost = g_cost + h_cost
                    node_priority_queue.push(new_node, f_cost)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
