#   Look for #IMPLEMENT tags in this file. These tags indicate what has
#   to be implemented to complete the warehouse domain.

#   You may add only standard python imports
#   You may not remove any imports.
#   You may not import or otherwise source any of your own files

import os  # for time functions
import math
from typing_extensions import runtime  # for infinity
from search import *  # for search engines
from sokoban import sokoban_goal_state, SokobanState, Direction, PROBLEMS  # for Sokoban specific classes and problems
import numpy as np

# SOKOBAN HEURISTICS
def heur_alternate(state):
    # IMPLEMENT
    '''a better heuristic'''
    '''INPUT: a sokoban state'''
    '''OUTPUT: a numeric value that serves as an estimate of the distance of the state to the goal.'''
    # heur_manhattan_distance has flaws.
    # Write a heuristic function that improves upon heur_manhattan_distance to estimate distance between the current state and the goal.
    # Your function should return a numeric value for the estimate of the distance to the goal.
    # EXPLAIN YOUR HEURISTIC IN THE COMMENTS. Please leave this function (and your explanation) at the top of your solution file, to facilitate marking.
    # Below is the implementation of the hueristic function
    # This implementation uses the larger addition part of the manhattan distance as the box-storage heuristic.
    # i.e., if a storage is (a_1, b_1), box is (a_2, b_2), then the heuristic is max(abs(a_1-a_2), abs(b_1-b_2))
    # We detect the deadlock if any two connected edges of a box has a wall, obstacle, or box as the neighbour,
    # see detail implementation of the deadlock at the detect_deadlock function
    # Then we add the same concept matric as the robot-box heuristic, this time we save the deadlock detection.
    # Sum those two parts together, this is our heuristic function.
    boxes = state.boxes
    storage = state.storage
    sum_dist = 0
    # heuristic for boxes to storages
    for box in boxes:
        if box in storage:
            continue
        if (detect_deadlock(box, state)):
                return math.inf
        min_dist = math.inf
        for dest in storage:
            dist0 = abs(dest[0]-box[0])
            dist1 = abs(dest[1]-box[1])
            if dist0 > dist1:
                if min_dist > dist0:
                    min_dist = dist0
            else:
                if min_dist > dist1:
                    min_dist = dist1
        sum_dist += min_dist

    # heuristic from robot to boxes
    for robot in state.robots:
        min_dist = math.inf
        for box in boxes:
            # we minus 1 since the robot cannot be overlapped with the box
            dist0 = abs(robot[0]-box[0])-1
            dist1 = abs(robot[1]-box[1])-1
            if dist0 > dist1:
                if min_dist > dist0:
                    min_dist = dist0
            else:
                if min_dist > dist1:
                    min_dist = dist1
            if min_dist == 0:
                break
        sum_dist += min_dist
    return sum_dist


def detect_deadlock(box, state):
    '''This is a helper function that detects whether the input box is in a deadlock
    based on the input state. Note: I call it in a deadlock if the box is not in the storage,
    and any two connected edges of that box has wall or obstacle or box neighbours.
    Since in this case, the box cannot be pushed in any direction, and it is not in the storage. 
    Return True if the box is in a deadlock, False if not.
    '''
    width = state.width
    height = state.height
    obstacle = state.obstacles
    boxes = state.boxes
    if (box==(0, 0) or box==(0, width-1) or box==(height-1, 0) or box==(height-1, width-1)):
        return True
    elif (box[0]==0) or (box[0]==height-1):
        if ((box[0], box[1]-1) in obstacle) or ((box[0], box[1]+1) in obstacle) or ((box[0], box[1]-1) in boxes) or ((box[0], box[1]+1) in boxes):
            return True
    elif (box[1]==0) or (box[1]==width-1):
        if ((box[0]-1, box[1]) in obstacle) or ((box[0]+1, box[1]) in obstacle) or ((box[0]-1, box[1]) in boxes) or ((box[0]+1, box[1]) in boxes):
            return True
    return False
    


def heur_zero(state):
    '''Zero Heuristic can be used to make A* search perform uniform cost search'''
    return 0

def heur_manhattan_distance(state):
    # IMPLEMENT
    '''admissible sokoban puzzle heuristic: manhattan distance'''
    '''INPUT: a sokoban state'''
    '''OUTPUT: a numeric value that serves as an estimate of the distance of the state to the goal.'''
    # We want an admissible heuristic, which is an optimistic heuristic.
    # It must never overestimate the cost to get from the current state to the goal.
    # The sum of the Manhattan distances between each box that has yet to be stored and the storage point nearest to it is such a heuristic.
    # When calculating distances, assume there are no obstacles on the grid.
    # You should implement this heuristic function exactly, even if it is tempting to improve it.
    # Your function should return a numeric value; this is the estimate of the distance to the goal.
    # Get the box and storage positions
    boxes = state.boxes
    storage = state.storage
    # Calculate the manhattan_distance
    sum_manhattan = 0
    for box in boxes:
        min_dist = math.inf
        for dest in storage:
            manhattan_dist = abs(dest[0]-box[0]) + abs(dest[1]-box[1])
            if min_dist > manhattan_dist:
                min_dist = manhattan_dist
        sum_manhattan += min_dist
    return sum_manhattan

def fval_function(sN, weight):
    # IMPLEMENT
    """
    Provide a custom formula for f-value computation for Anytime Weighted A star.
    Returns the fval of the state contained in the sNode.

    @param sNode sN: A search node (containing a SokobanState)
    @param float weight: Weight given by Anytime Weighted A star
    @rtype: float
    """
    return sN.gval + weight*sN.hval

# SEARCH ALGORITHMS
def weighted_astar(initial_state, heur_fn, weight, timebound):
    # IMPLEMENT    
    '''Provides an implementation of weighted a-star, as described in the HW1 handout'''
    '''INPUT: a sokoban state that represents the start state and a timebound (number of seconds)'''
    '''OUTPUT: A goal state (if a goal is found), else False as well as a SearchStats object'''
    '''implementation of weighted astar algorithm'''
    searching = SearchEngine('custom')
    wrapped_fval_function = lambda sN: fval_function(sN, weight)
    searching.init_search(initState=initial_state, goal_fn=sokoban_goal_state, heur_fn=heur_fn, fval_function=wrapped_fval_function)
    return searching.search(timebound = timebound)

def iterative_astar(initial_state, heur_fn, weight=1, timebound=5):  # uses f(n), see how autograder initializes a search line 88
    # IMPLEMENT
    '''Provides an implementation of realtime a-star, as described in the HW1 handout'''
    '''INPUT: a sokoban state that represents the start state and a timebound (number of seconds)'''
    '''OUTPUT: A goal state (if a goal is found), else False as well as a SearchStats object'''
    '''implementation of iterative astar algorithm'''
    searching = SearchEngine('custom')
    wrapped_fval_function = lambda sN: fval_function(sN, weight)
    costbound = None
    searching.init_search(initState=initial_state, goal_fn=sokoban_goal_state, heur_fn=heur_fn, fval_function=wrapped_fval_function)
    runtime = timebound
    stime = os.times()[0]
    state, stats = searching.search(timebound = runtime, costbound = costbound)
    runtime = timebound - (os.times()[0]-stime)
    if state is not False:
        hval = heur_fn(state)
        costbound = (state.gval, hval, (state.gval+(weight*hval)))
    while (runtime > 0):
        weight = weight/(3)
        wrapped_fval_function = lambda sN: fval_function(sN, weight)
        searching.init_search(initState=initial_state, goal_fn=sokoban_goal_state, heur_fn=heur_fn, fval_function=wrapped_fval_function)
        new_state, new_stats = searching.search(timebound = runtime, costbound=costbound)
        if new_state is not False:
            state = new_state
            stats = new_stats
            hval = heur_fn(state)
            costbound = (state.gval, hval, (state.gval+(weight*hval)))
        runtime = timebound - (os.times()[0]-stime)
    return state, stats

def iterative_gbfs(initial_state, heur_fn, timebound=5):  # only use h(n)
    # IMPLEMENT
    '''Provides an implementation of anytime greedy best-first search, as described in the HW1 handout'''
    '''INPUT: a sokoban state that represents the start state and a timebound (number of seconds)'''
    '''OUTPUT: A goal state (if a goal is found), else False'''
    '''implementation of iterative gbfs algorithm'''
    searching = SearchEngine('best_first')
    costbound = None
    searching.init_search(initState=initial_state, goal_fn=sokoban_goal_state, heur_fn=heur_fn, fval_function=None)
    runtime = timebound
    stime = os.times()[0]
    state, stats = searching.search(timebound = runtime, costbound = costbound)
    runtime = timebound - (os.times()[0]-stime)
    if state is not False:
        costbound = (state.gval, math.inf, math.inf)
    while (runtime > 0):
        new_state, new_stats = searching.search(timebound = runtime, costbound=costbound)
        if new_state is not False:
            state = new_state
            stats = new_stats
            costbound = (state.gval, math.inf, math.inf)
        runtime = timebound - (os.times()[0]-stime)
    return state, stats


