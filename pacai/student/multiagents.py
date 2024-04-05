import random
import math
from pacai.agents.base import BaseAgent
from pacai.agents.search.multiagent import MultiAgentSearchAgent
from pacai.core.distance import manhattan

class ReflexAgent(BaseAgent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.
    You are welcome to change it in any way you see fit,
    so long as you don't touch the method headers.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        `ReflexAgent.getAction` chooses among the best options according to the evaluation function.

        Just like in the previous project, this method takes a
        `pacai.core.gamestate.AbstractGameState` and returns some value from
        `pacai.core.directions.Directions`.
        """

        # Collect legal moves.
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions.
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current `pacai.bin.pacman.PacmanGameState`
        and an action, and returns a number, where higher numbers are better.
        Make sure to understand the range of different values before you combine them
        in your evaluation function.
        """

        successorGameState = currentGameState.generatePacmanSuccessor(action)

        # Useful information you can extract.
        newPosition = successorGameState.getPacmanPosition()
        oldFood = currentGameState.getFood()
        foodCoordinates = oldFood.asList()
        newGhostStates = successorGameState.getGhostStates()
        newCapsuleStates = successorGameState.getCapsules()
        newScaredTimes = [ghostState.getScaredTimer() for ghostState in newGhostStates]

        evaluation_score = currentGameState.getScore()
        
        foodDistances = []

        for foodPosition in foodCoordinates:
            foodDistance = manhattan(newPosition, foodPosition)
            foodDistances.append(foodDistance)

        if len(foodDistances) > 0:
            minfoodDistance = min(foodDistances)
        else:
            minfoodDistance = 0
        
        evaluation_score += minfoodDistance*-1.5

        ghostDistances = []
        
        ghost_times = []
        closest_ghost = 0

        for ghost, ghost_time in zip(newGhostStates, newScaredTimes):
             ghostPosition = tuple(map(int, ghost.getPosition()))
             ghostdist = manhattan(newPosition, ghostPosition)
             ghostDistances.append(ghostdist)
             ghost_times.append(ghost_time)

        if len(ghostDistances) > 0:
            closest_ghost_index = ghostDistances.index(min(ghostDistances))
            closest_ghost = ghostDistances[closest_ghost_index]
            closest_ghost_time = ghost_times[closest_ghost_index]
            if closest_ghost >= 1 and closest_ghost_time >= 2:
                evaluation_score += closest_ghost*2
            elif closest_ghost >= 1 and closest_ghost_time < 2:
                evaluation_score += (1/closest_ghost)*-2
            else:
                return -1000000

        if (closest_ghost > 0):
            evaluation_score += (1/closest_ghost)*-2
        else:
            evaluation_score += 0

        capsuleDistances = []

        for capsule in newCapsuleStates:
            capsuleDistances.append(manhattan(newPosition, capsule))

        if len(capsuleDistances) > 0:
            closest_capsule = min(capsuleDistances)
            evaluation_score += closest_capsule*-2
        else:
            evaluation_score += 100

        evaluation_score += 200 * sum(newScaredTimes)

        # print("final evalutation score: " + str(evaluation_score))
        return evaluation_score

class MinimaxAgent(MultiAgentSearchAgent):
    """
    A minimax agent.

    Here are some method calls that might be useful when implementing minimax.

    `pacai.core.gamestate.AbstractGameState.getNumAgents()`:
    Get the total number of agents in the game

    `pacai.core.gamestate.AbstractGameState.getLegalActions`:
    Returns a list of legal actions for an agent.
    Pacman is always at index 0, and ghosts are >= 1.

    `pacai.core.gamestate.AbstractGameState.generateSuccessor`:
    Get the successor game state after an agent takes an action.

    `pacai.core.directions.Directions.STOP`:
    The stop direction, which is always legal, but you may not want to include in your search.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        # Collect legal moves.
        legalMoves = gameState.getLegalActions()
        if ('Stop' in legalMoves):
            legalMoves.remove('Stop')
        # Choose one of the best actions.
        states = [gameState.generateSuccessor(0, action) for action in legalMoves]
        scores = [self.minmax(state, 0, 0) for state in states] #
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore] # gives you indices of best score
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.
        return legalMoves[chosenIndex]

    def minmax(self, gameState, agent, eval_depth):
        if (gameState.isOver() or (eval_depth == self.getTreeDepth())):
            return self.getEvaluationFunction()(gameState)
        Pacman = 0
        if (agent == Pacman):
            return self.maxAgent(gameState, agent, eval_depth)
        else:
            return self.minAgent(gameState, agent, eval_depth)
        
    def maxAgent(self, gameState, agent, eval_depth):
        max_eval = (math.inf)*-1
        legalMoves = gameState.getLegalActions(agent)
        if ('Stop' in legalMoves):
            legalMoves.remove('Stop')
        for action in legalMoves:
            successorState = gameState.generateSuccessor(agent,action)
            score = self.minmax( successorState, agent+1, eval_depth)
            if (score > max_eval):
                max_eval = score
        return max_eval

    def minAgent(self, gameState, agent, eval_depth):
        min_eval = math.inf
        legalMoves = gameState.getLegalActions(agent)
        if ('Stop' in legalMoves):
            legalMoves.remove('Stop')
        lastGhost = (agent+1)% gameState.getNumAgents() == 0
        if (lastGhost):
            eval_depth+=1
        for action in legalMoves:
            successorState = gameState.generateSuccessor(agent,action)
            if (lastGhost):
                score = self.minmax(successorState, 0, eval_depth)
            else:
                score = self.minmax(successorState, agent+1, eval_depth)
            if (score < min_eval):
                min_eval = score
        return min_eval

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    A minimax agent with alpha-beta pruning.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        # Collect legal moves.
        legalMoves = gameState.getLegalActions()
        if ('Stop' in legalMoves):
            legalMoves.remove('Stop')
        # Choose one of the best actions.
        a = (math.inf)*-1
        b = math.inf
        states = [gameState.generateSuccessor(0, action) for action in legalMoves]
        scores = [self.ABminmax(state, 0, 0, a, b) for state in states] #
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore] # gives you indices of best score
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.

        return legalMoves[chosenIndex]

    def ABminmax(self, gameState, agent, eval_depth, Alpha, Beta):
        if (gameState.isOver() or (eval_depth == self.getTreeDepth())):
            return self.getEvaluationFunction()(gameState)
        Pacman = 0
        if (agent == Pacman):
            return self.maxAgent(gameState, agent, eval_depth, Alpha, Beta)
        else:
            return self.minAgent(gameState, agent, eval_depth, Alpha, Beta)
        
    def maxAgent(self, gameState,  agent, eval_depth, Alpha, Beta):
        max_eval = (math.inf)*-1
        legalMoves = gameState.getLegalActions(agent)
        if ('Stop' in legalMoves):
            legalMoves.remove('Stop')
        for action in legalMoves:
            successorState = gameState.generateSuccessor(agent,action)
            score = self.ABminmax( successorState, agent+1, eval_depth, Alpha, Beta)
            if (score > max_eval):
                max_eval = score
            if (score > Alpha):
                Alpha = score
            if (Alpha >= Beta):
                break
        return max_eval

    def minAgent(self, gameState, agent, eval_depth, Alpha, Beta):
        min_eval = math.inf
        legalMoves = gameState.getLegalActions(agent)
        if ('Stop' in legalMoves):
            legalMoves.remove('Stop')
        lastGhost = (agent+1)% gameState.getNumAgents() == 0
        if (lastGhost):
            eval_depth+=1
        for action in legalMoves:
            successorState = gameState.generateSuccessor(agent,action)
            if (lastGhost):
                score = self.ABminmax(successorState, 0, eval_depth, Alpha, Beta)
            else:
                score = self.ABminmax( successorState, agent+1, eval_depth, Alpha, Beta)
            if (score < min_eval):
                min_eval = score
            if (score < Beta):
                Beta = score
            if (Alpha >= Beta):
                break
        return min_eval

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    An expectimax agent.

    All ghosts should be modeled as choosing uniformly at random from their legal moves.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the expectimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        # Collect legal moves.
        legalMoves = gameState.getLegalActions()
        if ('Stop' in legalMoves):
            legalMoves.remove('Stop')
        # Choose one of the best actions.
        states = [gameState.generateSuccessor(0, action) for action in legalMoves]
        scores = [self.expectimax(state, 0, 0) for state in states] #
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore] # gives you indices of best score
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.
        return legalMoves[chosenIndex]
    
    def expectimax(self, gameState, agent, eval_depth):
        if (gameState.isOver() or (eval_depth == self.getTreeDepth())):
            return self.getEvaluationFunction()(gameState)
        Pacman = 0
        if (agent == Pacman):
            return self.maxAgent(gameState, agent, eval_depth)
        else:
            return self.expectiAgent(gameState, agent, eval_depth)

    def maxAgent(self, gameState, agent, eval_depth):
        max_eval = (math.inf)*-1
        legalMoves = gameState.getLegalActions(agent)
        if ('Stop' in legalMoves):
            legalMoves.remove('Stop')
        for action in legalMoves:
            successorState = gameState.generateSuccessor(agent,action)
            score = self.expectimax( successorState, agent+1, eval_depth)
            if (score > max_eval):
                max_eval = score
        return max_eval
        
    def expectiAgent(self, gameState, agent, eval_depth):
        expecti_eval = 0
        legalMoves = gameState.getLegalActions(agent)
        if ('Stop' in legalMoves):
            legalMoves.remove('Stop')
        distrution = 1.0/len(legalMoves)
        lastGhost = (agent+1)% gameState.getNumAgents() == 0
        if (lastGhost):
            eval_depth+=1
        for action in legalMoves:
            successorState = gameState.generateSuccessor(agent, action)
            if (lastGhost):
                expecti_eval += self.expectimax(successorState, 0, eval_depth) * distrution
            else:
                expecti_eval += self.expectimax(successorState, agent+1, eval_depth) * distrution
        return expecti_eval

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable evaluation function.

    DESCRIPTION: 
    """
        # Useful information extracted.
    Position = currentGameState.getPacmanPosition()
    Food = currentGameState.getFood()
    foodCoordinates = Food.asList()
    GhostStates = currentGameState.getGhostStates()
    CapsuleStates = currentGameState.getCapsules()
    ScaredTimes = [ghostState.getScaredTimer() for ghostState in GhostStates]

    evaluation_score = currentGameState.getScore()
        
    foodDistances = []
    for foodPosition in foodCoordinates:
        foodDistance = manhattan(Position, foodPosition)
        foodDistances.append(foodDistance)

    if len(foodDistances) > 0:
        minfoodDistance = min(foodDistances)
    else:
        minfoodDistance = 0
        
    evaluation_score += minfoodDistance*-1.5 # evaulation score based on the distance to the cloest food 

    ghostDistances = []
        
    ghost_times = []
    closest_ghost = 0

    for ghost, ghost_time in zip(GhostStates, ScaredTimes): # for each ghost and their indivual scare timers
         
         # find the position and distnaces of eacn ghost and the amount of time they have left on their timer
         ghostPosition = tuple(map(int, ghost.getPosition()))
         ghostdist = manhattan(Position, ghostPosition)
         ghostDistances.append(ghostdist)
         ghost_times.append(ghost_time)

    if len(ghostDistances) > 0: # if not touching ghost
        closest_ghost_index = ghostDistances.index(min(ghostDistances))
        closest_ghost = ghostDistances[closest_ghost_index]
        closest_ghost_time = ghost_times[closest_ghost_index]
        if closest_ghost >= 1 and closest_ghost_time >= 2: # the evulation score is better if we are closer to a ghost when the timer is >=2
            evaluation_score += closest_ghost*2
        elif closest_ghost >= 1 and closest_ghost_time < 2: # encourages the agent to move away from the ghost if the timer is < 2
            evaluation_score += (1/closest_ghost)*-2
        else:
            return -1000000

    capsuleDistances = []

    for capsule in CapsuleStates:
        capsuleDistances.append(manhattan(Position, capsule))

    if len(capsuleDistances) > 0: 
        minCapsuleDistance = min(capsuleDistances)
    else:
        minCapsuleDistance = 0


    evaluation_score += minCapsuleDistance*-2 # encourages the agent to move towards the capsule

    return evaluation_score

class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest.

    You can use any method you want and search to any depth you want.
    Just remember that the mini-contest is timed, so you have to trade off speed and computation.

    Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
    just make a beeline straight towards Pacman (or away if they're scared!)

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
