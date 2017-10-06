# myTeam.py
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


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
import json
import ast
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveAgent', second = 'DefensiveReflexAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

# Q-Learning
class QAgent(CaptureAgent):

    # alpha: learning rate
    # gamma: discount
    # epsilon: exploration rate
    def __init__(self, index, episodeCount = 1000, alpha = 0.2, gamma = 0.85, epsilon = 0.05, path = "table.txt"):

        print "Q init"

        CaptureAgent.__init__(self, index)

        self.episodeCount = episodeCount
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.path = path
        self.startPos = None
        self.lastAction = None

        # using load here when tranning
        # it is linear Agent when having loadW
        # !!!!!!!!!!!
        # COMMENT following 4 lines for contest
        if hasattr(self, 'loadW'):
            self.loadW()
        else:
            self.loadQ()


    def loadQ(self):

        self.q = util.Counter()

        filePath = "classical-" + self.path

        with open(filePath) as f:
            tmp = json.load(f)
            for k,v in tmp.iteritems():
                self.q[ast.literal_eval(k)] = v

    def saveQ(self):

        filePath = "classical-" + self.path

        print "Saving Q to " + filePath + "..."

        with open(filePath, 'w') as f:
            json.dump({str(k): v for k, v in self.q.iteritems()}, f)

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the
        agent to populate useful fields (such as what team
        we're on).

        A distanceCalculator instance caches the maze distances
        between each pair of positions, so your agents can use:
        self.distancer.getDistance(p1, p2)

        IMPORTANT: This method may run for at most 15 seconds.
        """

        '''
        Make sure you do not delete the following line. If you would like to
        use Manhattan distances instead of maze distances in order to save
        on initialization time, please take a look at
        CaptureAgent.registerInitialState in captureAgents.py.
        '''
        CaptureAgent.registerInitialState(self, gameState)

        '''
        Your initialization code goes here, if you need any.
        '''

        self.initialFood  = self.getFood(gameState).count()
        self.initialMyfood = self.getFoodYouAreDefending(gameState).count()

        self.cachedDistance = {}

        # !!!!!!!!!!!
        # 15s Limitation
        # For contest, UNCOMMENT self.loadQ Here !!!!!!!!!!!!!!!!!!!
        # !!!!!!!!!!!!

        # self.path = "classical-" + self.path
        # self.loadQ()



    # return q(s,a)
    def getQValue(self, features):

        return self.q[features]

    # return max Q, based on a list of (s,a)
    def getQMax(self, qKeys):

        return max(self.getQValue(key[0]) for key in qKeys)

    # return a list of (features, action)
    # since self.getFeatures might take a lot of time, it is
    # important to call a stand-alone getQKeys and store this value
    def getQKeys(self, state):

        actions = state.getLegalActions(self.index)

        return [(self.getFeatures(state, action), action) for action in actions]

    def chooseAction(self, gameState):

        if self.startPos is None:
            self.startPos = gameState.getAgentPosition(self.index)

        lastState = self.getPreviousObservation()

        # update Q
        if (lastState is not None and self.lastAction is not None):
            reward = self.determineReward(lastState, self.lastAction, gameState)
            self.update(lastState, self.lastAction, gameState, reward)

        action = self.determineAction(gameState)

        self.lastAction = action

        return action

    def determineAction(self, state):

        # epsilon-greedy explore
        if (util.flipCoin(self.epsilon)):
            return random.choice(state.getLegalActions(self.index))

        qKeys = self.getQKeys(state)

        maxQ = self.getQMax(qKeys)

        bestActions = [key[1] for key in qKeys if self.getQValue(key[0]) == maxQ]

        #best = [key for key in qKeys if self.getQValue(key[0]) == maxQ]

        #print best, maxQ

        return random.choice(bestActions)

    def update(self, oldState, action, newState, reward):

        oldFeature = self.getFeatures(oldState, action)

        oldQ = self.getQValue(oldFeature)
        newQ = self.getQMax(self.getQKeys(newState))

        self.q[oldFeature] = (oldQ +
            self.alpha * (reward + self.gamma * newQ - oldQ))

    # this method must be overrided in subclass
    def determineReward(self, oldState, oldAction, newState):

        util.raiseNotDefined()

    # this method must be overrided in subclass
    def getFeatures(self, state, action):

        util.raiseNotDefined()

    def getSuccessor(self, gameState, action):

        successor = gameState.generateSuccessor(self.index, action)

        pos = successor.getAgentState(self.index).getPosition()
        if pos != util.nearestPoint(pos):
          # Only half a grid position was covered ??
          return successor.generateSuccessor(self.index, action)
        else:
          return successor

    # pos1 for food
    def quickGetDistance(self, pos1, pos2):

        if pos1 == pos2:
            return 0

        key = (pos1, pos2)

        if key not in self.cachedDistance:
            self.cachedDistance[key] = self.getMazeDistance(pos1, pos2)

        return self.cachedDistance[key]

    def isScared(self, gameState, index):

        scared = gameState.data.agentStates[index].scaredTimer <= 0
        return scared



class LinearQAgent(QAgent):

    def registerInitialState(self, gameState):

        CaptureAgent.registerInitialState(self, gameState)

        self.lastAction = None
        self.cachedDistance = {}

        # !!!!!!!!!!!
        # 15s Limitation
        # For contest, UNCOMMENT self.loadW Here !!!!!!!!!!!!!!!!!!!
        # !!!!!!!!!!!!

        # self.loadW()

    def loadW(self):

        self.w = util.Counter()

        filePath = "linear-" + self.path

        with open(filePath) as f:
            tmp = json.load(f)
            for k,v in tmp.iteritems():
                self.w[k] = v

    def saveW(self):

        filePath = "linear-" + self.path

        print "Saving W to " + filePath + "..."

        with open(filePath, 'w') as f:
            json.dump({k: v for k, v in self.w.iteritems()}, f)

    def getQValue(self, features):

        q = 0;

        for k, v in features.iteritems():
            q += self.w[k] * v

        return q

    def update(self, oldState, action, newState, reward):

        oldFeature = self.getFeatures(oldState, action)

        oldQ = self.getQValue(oldFeature)
        newQ = self.getQMax(self.getQKeys(newState))

        for k, v in oldFeature.iteritems():
            self.w[k] += self.alpha * (reward + self.gamma * newQ - oldQ) * v




class OffensiveAgent(QAgent):

    def __init__(self, index, episodeCount = 1000, alpha = 0.2, gamma = 0.8, epsilon = 0.05, path = "offense.json"):

        QAgent.__init__(self, index, episodeCount, alpha, gamma, epsilon, path)


    def getFeatures(self, state, action):

        successor = self.getSuccessor(state, action)
        foodlist = self.getFood(state).asList()
        capsules = self.getCapsules(state)

        nextP = successor.getAgentState(self.index).getPosition()
        nowP = state.getAgentState(self.index).getPosition()

        opponents = [state.getAgentState(i) for i in self.getOpponents(state)]

        closeEnemys = []

        # cannot see enemies who are 6 steps away
        # using this feature might not get ideal results
        # nearest enemy's distance in next Step
        nearestEnemyDistance = 6

        for opponent in opponents:
            if opponent and not opponent.isPacman and opponent.getPosition() is not None and opponent.scaredTimer <= 0:
                distance = self.quickGetDistance(opponent.getPosition(), nextP)
                if distance <= 1:
                    closeEnemys.append(opponent)
                if distance < nearestEnemyDistance:
                    nearestEnemyDistance = distance


        numCloseEnemys = len(closeEnemys)
        canEatFood = canEatCapsule = 0
        if numCloseEnemys == 0 :
            if nextP in foodlist:
                canEatFood = 1
            if nextP in capsules:
                canEatCapsule = 1


        minFoodDistance = min([self.quickGetDistance(food, nextP) for food in foodlist])
        numNextLegalAction = len(successor.getLegalActions(self.index))

        numCarrying = state.data.agentStates[self.index].numCarrying

        #isRed = int(self.red)
        isPacman = int(state.getAgentState(self.index).isPacman)

        # TODO modify, as this is not exactly to check if the agent is moving to base
        isMovingToBase = int(self.quickGetDistance(self.startPos, nextP) < self.quickGetDistance(self.startPos, nowP))

        # it's linear q agent
        if issubclass(self.__class__, LinearQAgent):

            features = util.Counter()

            features["bias"] = 1.0
            features["numCloseEnemys"] = numCloseEnemys
            features["nearestEnemyDistance"] = nearestEnemyDistance
            features["canEatFood"] = canEatFood
            features["minFoodDistance"] = canEatFood
            features["numNextLegalAction"] = numNextLegalAction
            features["numCarrying"] = numCarrying
            features["isPacman"] = isPacman
            features["isMovingToBase"] = isMovingToBase
            #features["isRed"] = isRed

        else:
        # it's classical q agent

            # roughly 3 * 7 * 2 * 2 * 100 * 5 * 30 * 2 ~ 1M
            features = (numCloseEnemys, nearestEnemyDistance, canEatFood, canEatCapsule,
                minFoodDistance, numNextLegalAction, numCarrying, isPacman, isMovingToBase) #isRed)

        return features

    def determineReward(self, oldState, oldAction, newState):

        reward = 0

        nowPos = newState.getAgentPosition(self.index)
        lastPos =  oldState.getAgentPosition(self.index)

        diff = (newState.data.agentStates[self.index].numReturned -
            oldState.data.agentStates[self.index].numReturned)

        # bring food back
        if (diff > 0):
            print "bring food back!!!"
            reward = diff * 20
        else:
            # eat new food
            if newState.data.agentStates[self.index].numCarrying > oldState.data.agentStates[self.index].numCarrying:
                print "eat new food"
                reward = 5
            # get eaten by enemy as a pacman
            # currently we dont punish the offensive agent when it gets eaten as a scared ghost
            # cuz that would make it try to avoid enemy even in our side (which is wrong in most cases)
            elif nowPos == self.startPos and oldState.data.agentStates[self.index].isPacman:
                reward = -100 + oldState.data.agentStates[self.index].numCarrying * (-10)
            else:

                # eat capsule
                if nowPos in self.getCapsules(oldState):
                    print "eat capsule"
                    reward = 100
                # walking in the enemy's side (i.e. offensive)
                elif newState.data.agentStates[self.index].isPacman:
                    print "walking in the enemy's side"
                    reward = -1
                # get closer to the nearest food
                else:
                    # compared to the foodlist of oldState
                    foodlist = self.getFood(oldState).asList()
                    nowMinFoodDistance = min([self.quickGetDistance(food, nowPos) for food in foodlist])
                    lastMinFoodDistance = min([self.quickGetDistance(food, lastPos) for food in foodlist])
                    if (nowMinFoodDistance < lastMinFoodDistance):
                        reward = -1.5
                    else:
                        print "moving backward"
                        reward = -3

                """
                # walking in own side, towards the enemy's side (i.e. offensive)
                elif (self.red and oldAction == Directions.EAST) or (not self.red and oldAction == Directions.WEST):
                    reward = -1
                # stop
                elif oldAction == Directions.STOP:
                    reward = -3
                else:
                    reward = -2
                """
        return reward

class DefensiveAgent(QAgent):

    def __init__(self, index, episodeCount = 1000, alpha = 0.2, gamma = 0.8, epsilon = 0.05, path = "defense.json"):

        QAgent.__init__(self, index, episodeCount, alpha, gamma, epsilon, path)


    def getEnemyPosition(self, state):
        lastState = self.getPreviousObservation()
        currentFoodList = self.getFoodYouAreDefending(state).asList()
        currentCap = self.getCapsulesYouAreDefending(state).asList()


        opponents = [state.getAgentState(i) for i in self.getOpponents(state)]
        opponentsP = []
        for opponent in opponents:
            if opponent and opponent.isPacman and opponent.getPosition():
                opponentsP.append(opponent)

        if lastState:
            lastFoodList = self.getFoodYouAreDefending(lastState).asList()
            lastCap = self.getCapsulesYouAreDefending(lastState).asList()

            foodEaten = set(currentFoodList) - set(lastFoodList)
            capEaten = set(currentCap) - set(lastCap)
            if foodEaten:
                for food in foodEaten:
                    if food not in opponentsP:
                        opponentsP.append(food)
            if capEaten:
                for cap in capEaten:
                    if cap not in opponentsP:
                        opponentsP.append(cap)

        return opponentsP

    def minDistanceToEnemy(self, state, position):
        opponents = [state.getAgentState(i) for i in self.getOpponents(state)]

        # Min Distance to Enemy
        minDisToEnemy = 999999
        for opponent in opponents:
            if opponent and opponent.isPacman and opponent.getPosition() is not None:
                d = self.quickGetDistance(opponent.getPosition(), position)
                if d < minDisToEnemy:
                    minDisToEnemy = d
        return minDisToEnemy

    def getfarestFood(self, state):
        #Get the farest food I can protect
        # May be enemy's target
        myFoods = self.getFoodYouAreDefending(state).asList()
        farestFood = None
        maxX = -1
        minX = 999999
        for myfood in myFoods:
            if state.isOnRedTeam(self.index):
                if myfood[0] > maxX:
                    farestFood = myfood
            else:
                if myfood[0] < minX:
                    farestFood = myfood
        return farestFood

    def getFeatures(self, state, action):

        #features = util.Counter()
        myFoods = self.getFoodYouAreDefending(state).asList()
        foodlist = self.getFood(state).asList()

        successor = self.getSuccessor(state, action)
        nextP = successor.getAgentState(self.index).getPosition()

        minDisToEnemy = self.minDistanceToEnemy(state, nextP)

        #Is Scared
        scared = 1 if self.isScared(state,self.index) else 0

        # No of Next state's actions
        numNextLegalAction = len(successor.getLegalActions(self.index))

        ############# Newly Added
        farestFood = self.getfarestFood(state)
        if farestFood:
            minDistToSon = self.quickGetDistance(nextP, farestFood)

        minDisToDot = min([self.quickGetDistance(nextP, food) for food in myFoods])

        numOfMySons = len(myFoods) / self.initialMyfood
        numofFoodtoeat = len(foodlist) / self.initialFood
        ##############

        # its linear q agent
        if issubclass(self.__class__, LinearQAgent):
            features = None
        else:
            features = (minDisToEnemy, numNextLegalAction, scared)

        return features


            #todo
    def determineReward(self, oldState, oldAction, newState):
        currentP         = newState.getAgentPosition(self.index)
        lastP            = oldState.getAgentPosition(self.index)
        currentOpponents = [newState.getAgentState(i) for i in self.getOpponents(newState)]
        pastOpponents    = [oldState.getAgentState(i) for i in self.getOpponents(oldState)]

        minCurrentDisToOp = self.minDistanceToEnemy(newState, currentP)
        minLastDisToOp    = self.minDistanceToEnemy(oldState, lastP)

        #Default, nothing special move
        reward = -1

        # Cross border
        if newState.data.agentStates[self.index].isPacman:
            reward -= 100
        # No move
        if lastP == currentP :
            reward -= 10

        myFoods = self.getFoodYouAreDefending(newState).asList()
        farestFood = self.getfarestFood(newState)
        # Close to farest food
        if farestFood:
            if self.quickGetDistance(currentP, farestFood) <= 5:
                reward += 5
        if self.quickGetDistance(lastP, farestFood) >= self.quickGetDistance(currentP, farestFood):
            reward += 1

        meetEnemy = False
        for opponent in currentOpponents:
            if opponent and opponent.isPacman and opponent.getPosition() is not None:
                if newState.getAgentPosition(self.index) == opponent.getPosition():
                    meetEnemy = True

        # meetEnemy = newState.getAgentPosition(self.index) in currentOpponents
        if self.isScared(newState, self.index):
            # Eaten
            if meetEnemy:
                reward -= 200
            # Chasing
            elif minCurrentDisToOp >= minLastDisToOp and minLastDisToOp != 999999:
                reward += 20
        else:
            #Eat enemy
            if meetEnemy:
                reward += 500

        return reward


class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """

  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

class OffensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()
    features['successorScore'] = -len(foodList)#self.getScore(successor)

    # Compute distance to the nearest food

    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -1}

class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}
