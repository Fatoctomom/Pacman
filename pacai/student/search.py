"""
In this file, you will implement generic search algorithms which are called by Pacman agents.
"""
from pacai.util.queue import Queue
from pacai.util.stack import Stack
from pacai.util.priorityQueue import PriorityQueue

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first [p 85].

    Your search algorithm needs to return a list of actions that reaches the goal.
    Make sure to implement a graph search algorithm [Fig. 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    ```
    print("Start: %s" % (str(problem.startingState())))
    print("Is the start a goal?: %s" % (problem.isGoal(problem.startingState())))
    print("Start's successors: %s" % (problem.successorStates(problem.startingState())))
    ```
    """

    node = problem.startingState()
    reached = set()
    actions = []
    if (problem.isGoal(node)):
        return actions
    fringe = Stack()
    fringe.push((node, actions))

    while (not fringe.isEmpty()):
        current_state, current_path = fringe.pop()

        if problem.isGoal(current_state):
            return current_path

        if (current_state in reached):
            continue

        reached.add(current_state)

        for child in problem.successorStates(current_state):
            childstate = child[0]
            action = child[1]
            if childstate not in reached and childstate not in fringe.list:
                fringe.push((childstate, current_path + [action]))
    return None

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first. [p 81]
    """

    node = problem.startingState()
    reached = set()
    actions = []
    current_path = []
    if (problem.isGoal(node)):
        return actions
    fringe = Queue()
    fringe.push((node, actions))

    while (not fringe.isEmpty()):
        current_state, current_path = fringe.pop()

        if problem.isGoal(current_state):
            return current_path

        if (current_state in reached):
            continue

        reached.add(current_state)

        for child in problem.successorStates(current_state):
            childstate = child[0]
            action = child[1]
            if childstate not in reached and childstate not in fringe.list:
                if isinstance(action, list):
                    fringe.push((childstate, current_path + action))
                else:
                    fringe.push((childstate, current_path + [action]))
    return None


def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """

    node = problem.startingState()
    reached = set()
    actions = []
    path_cost = 0
    fringe = PriorityQueue()
    fringe.push((node, actions, path_cost), 0)

    while (not fringe.isEmpty()):
        current_state, current_path, current_cost = fringe.pop()

        if problem.isGoal(current_state):
            return current_path

        if (current_state in reached):
            continue

        reached.add(current_state)

        for child in problem.successorStates(current_state):
            childstate = child[0]
            action = child[1]
            cost = child[2]
            if childstate not in reached and childstate not in fringe.heap:
                path_cost = current_cost + cost
                fringe.push((childstate, current_path + [action], path_cost), path_cost)
    return None

def aStarSearch(problem, heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """

    node = problem.startingState()
    reached = set()
    actions = []
    path_cost = 0
    estimated_value = estimate(node, path_cost, heuristic, problem)
    
    fringe = PriorityQueue()
    fringe.push((node, actions, path_cost), estimated_value)

    while (not fringe.isEmpty()):
        current_state, current_path, current_cost = fringe.pop()

        if problem.isGoal(current_state):
            return current_path

        if (current_state in reached):
            continue

        reached.add(current_state)

        for child in problem.successorStates(current_state):
            childstate = child[0]
            action = child[1]
            cost = child[2]
            if childstate not in reached and childstate not in fringe.heap:
                path_cost = current_cost + cost
                estimated_value = estimate(childstate, path_cost, heuristic, problem)
                fringe.push((childstate, current_path + [action], path_cost), estimated_value)
    return None


def estimate(current_state, path_cost, heurstic, problem):
    # calculate the heurstic
    heurstic_value = heurstic(current_state, problem)

    # add to current path cost
    ret = path_cost + heurstic_value

    # G(n)
    return ret
