"""
This file contains incomplete versions of some agents that can be selected to control Pacman.
You will complete their implementations.

Good luck and happy searching!
"""

import logging

from pacai.util.queue import Queue
from pacai.core.actions import Actions
from pacai.core.search.position import PositionSearchProblem
from pacai.core.search.problem import SearchProblem
from pacai.agents.base import BaseAgent
from pacai.agents.search.base import SearchAgent
from pacai.core.directions import Directions
from pacai.core.distance import manhattan

class CornersProblem(SearchProblem):
    """
    This search problem finds paths through all four corners of a layout.

    You must select a suitable state space and successor function.
    See the `pacai.core.search.position.PositionSearchProblem` class for an example of
    a working SearchProblem.

    Additional methods to implement:

    `pacai.core.search.problem.SearchProblem.startingState`:
    Returns the start state (in your search space,
    NOT a `pacai.core.gamestate.AbstractGameState`).

    `pacai.core.search.problem.SearchProblem.isGoal`:
    Returns whether this search state is a goal state of the problem.

    `pacai.core.search.problem.SearchProblem.successorStates`:
    Returns successor states, the actions they require, and a cost of 1.
    The following code snippet may prove useful:
    ```
        successors = []

        for action in Directions.CARDINAL:
            x, y = currentPosition
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            hitsWall = self.walls[nextx][nexty]

            if (not hitsWall):
                # Construct the successor.

        return successors
    ```
    """

    def __init__(self, startingGameState):
        super().__init__()

        self.walls = startingGameState.getWalls()
        self.startingPosition = startingGameState.getPacmanPosition()
        top = self.walls.getHeight() - 2
        right = self.walls.getWidth() - 2

        self.corners = ((1, 1), (1, top), (right, 1), (right, top))
        self.neededcorners = self.corners
        for corner in self.corners:
            if not startingGameState.hasFood(*corner):
                logging.warning('Warning: no food in corner ' + str(corner))

        if (self.startingPosition is None):
            raise ValueError("Could not find starting location.")
        
    def startingState(self):
        neededcorners = self.corners
        return (self.startingPosition, neededcorners)
    
    def isGoal(self, state):
        current_pos, needed_corners = state
        if (not needed_corners):
            return True
        else:
            return False
        
    def successorStates(self, state):
        current_pos, needed_corners = state
        if current_pos in needed_corners:
            tempset = set(needed_corners)
            tempset.discard(current_pos)
            needed_corners = tuple(tempset)
        successors = []
        for action in Directions.CARDINAL:
            x, y = current_pos
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            hitsWall = self.walls[nextx][nexty]

            if (not hitsWall):
                next_pos = (nextx, nexty)
                next_state = (next_pos, needed_corners)
                successors.append((next_state, action, 1))
        
        self._numExpanded += 1
        if (current_pos not in self._visitedLocations):
            self._visitedLocations.add(current_pos)
            # Note: visit history requires coordinates not states. In this situation
            # they are equivalent.
            coordinates = current_pos
            self._visitHistory.append(coordinates)

        return successors


    def actionsCost(self, actions):
        """
        Returns the cost of a particular sequence of actions.
        If those actions include an illegal move, return 999999.
        This is implemented for you.
        """

        if (actions is None):
            return 999999

        x, y = self.startingPosition
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999

        return len(actions)

def cornersHeuristic(state, problem):
    """
    A heuristic for the CornersProblem that you defined.

    This function should always return a number that is a lower bound
    on the shortest path from the state to a goal of the problem;
    i.e. it should be admissible.
    (You need not worry about consistency for this heuristic to receive full credit.)
    """

    # Useful information.
    # corners = problem.corners  # These are the corner coordinates
    # walls = problem.walls  # These are the walls of the maze, as a Grid.

    # *** Your Code Here ***
    ret = 0
    wallmaze = problem.walls
    corners = list(problem.corners)
    current_pos = state[0]
    distances = []
    while (len(corners) > 0):
        distances = []
        for corner in corners:
            distance = trueDist(wallmaze, current_pos, corner)
            distances.append((distance, corner))
        distances.sort(key = lambda x: x[0])
        mindist, mincorner = distances[0]
        ret += mindist
        current_pos = mincorner
        corners.remove(mincorner)
    return ret

def trueDist(maze, start, end):
    rows = maze.getHeight()
    cols = maze.getWidth()
    visted = [[False for _ in range(rows)] for _ in range(cols)]
    queue = Queue()
    queue.push((start, 0))
    while queue:
        (col, row), dist = queue.pop()
        if (col, row) == end:
            return dist
        for y, x in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            newy, newx = int(y + col), int(x + row)
            if not visted[newy][newx] and not maze[newy][newx]:
                visted[newy][newx] = True
                queue.push(((newy, newx), (dist + 1)))
    return -1

def foodHeuristic(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.
    First, try to come up with an admissible heuristic;
    almost all admissible heuristics will be consistent as well.

    If using A* ever finds a solution that is worse than what uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!
    On the other hand, inadmissible or inconsistent heuristics may find optimal solutions,
    so be careful.

    The state is a tuple (pacmanPosition, foodGrid) where foodGrid is a
    `pacai.core.grid.Grid` of either True or False.
    You can call `foodGrid.asList()` to get a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the problem.
    For example, `problem.walls` gives you a Grid of where the walls are.

    If you want to *store* information to be reused in other calls to the heuristic,
    there is a dictionary called problem.heuristicInfo that you can use.
    For example, if you only want to count the walls once and store that value, try:
    ```
    problem.heuristicInfo['wallCount'] = problem.walls.count()
    ```
    Subsequent calls to this heuristic can access problem.heuristicInfo['wallCount'].
    """

    current_pos, foodGrid = state

    food_coord = foodGrid.asList()
    distances = []
    distances_food = [0]
    wallmaze = problem.walls

    for food in food_coord:
        distances.append(trueDist(wallmaze, current_pos, food))
        for fooddist in food_coord:
            distances_food.append(trueDist(wallmaze, food, fooddist))

    if len(distances):
        mindist = min(distances)
        farthest_food = max(distances_food)
        return mindist + farthest_food
    else:
        return max(distances_food)

class ClosestDotSearchAgent(SearchAgent):
    """
    Search for all food using a sequence of searches.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def registerInitialState(self, state):
        self._actions = []
        self._actionIndex = 0

        currentState = state

        while (currentState.getFood().count() > 0):
            nextPathSegment = self.findPathToClosestDot(currentState)  # The missing piece
            self._actions += nextPathSegment

            for action in nextPathSegment:
                legal = currentState.getLegalActions()
                if action not in legal:
                    raise Exception('findPathToClosestDot returned an illegal move: %s!\n%s' %
                            (str(action), str(currentState)))

                currentState = currentState.generateSuccessor(0, action)

        logging.info('Path found with cost %d.' % len(self._actions))

    def findPathToClosestDot(self, gameState):
        """
        Returns a path (a list of actions) to the closest dot, starting from gameState.
        """

        # Here are some useful elements of the startState
        # startPosition = gameState.getPacmanPosition()
        # food = gameState.getFood()
        # walls = gameState.getWalls()
        # problem = AnyFoodSearchProblem(gameState)

        startPosition = gameState.getPacmanPosition()
        foodGrid = gameState.getFood()
        wallmaze = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState)
        food_coord = foodGrid.asList()

        distances = []
        for food in food_coord:
            distances.append(((trueDist(wallmaze, startPosition, food)), food))
        mindist = min(distances, key=lambda x: x[0])
        closestfood = mindist[1]

        reached = set()
        actions = []
        if (problem.isGoal(startPosition, closestfood)):
            return actions
        
        fringe = Queue()
        fringe.push((startPosition, actions))
        while (not fringe.isEmpty()):
            current_state, current_path = fringe.pop()

            if problem.isGoal(current_state, closestfood):
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

class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem,
    but has a different goal test, which you need to fill in below.
    The state space and successor function do not need to be changed.

    The class definition above, `AnyFoodSearchProblem(PositionSearchProblem)`,
    inherits the methods of `pacai.core.search.position.PositionSearchProblem`.

    You can use this search problem to help you fill in
    the `ClosestDotSearchAgent.findPathToClosestDot` method.

    Additional methods to implement:

    `pacai.core.search.position.PositionSearchProblem.isGoal`:
    The state is Pacman's position.
    Fill this in with a goal test that will complete the problem definition.
    """

    def __init__(self, gameState, start = None):
        super().__init__(gameState, goal = None, start = start)

        # Store the food for later reference.
        self.food = gameState.getFood()
        self.foodcoord = self.food.asList()

    def isGoal(self, state):
        current_pos = state
        if current_pos in self.foodcoord:
            return True
        else:
            return False

class ApproximateSearchAgent(BaseAgent):
    """
    Implement your contest entry here.

    Additional methods to implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Get a `pacai.bin.pacman.PacmanGameState`
    and return a `pacai.core.directions.Directions`.

    `pacai.agents.base.BaseAgent.registerInitialState`:
    This method is called before any moves are made.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
