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

import time

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'DumbDefensiveAgent', second = 'DumbDefensiveAgent'):
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


# a dumb defensive agent
class DumbDefensiveAgent(CaptureAgent):

    def registerInitialState(self, state):

        CaptureAgent.registerInitialState(self, state)

        self.walls = state.getWalls().data
        self.height = state.getWalls().height
        self.width = state.getWalls().width
        self.startPos = state.getAgentState(self.index).getPosition()
        self.enemyStartPos = [state.getAgentState(i).getPosition() for i in self.getOpponents(state)][0]

        self.cachedDistance = {}

        # last status, make sure update this in chooseAction
        # do not get these status based on self.getPreviousObservation
        self.lastMyFoods = self.getFoodYouAreDefending(state)
        self.lastTheirFoods = self.getFood(state)
        self.lastInvadingEnemyPos = []

        self.computeDeadEnds(state)

        """
        import sys
        #print the map
        for y in reversed(range(self.height)):
            for x in range(self.width):
                if self.walls[x][y]:
                    sys.stdout.write('#')
                if (x,y) in self.deadEnds:
                    sys.stdout.write('@')
                elif not self.walls[x][y]:
                    sys.stdout.write(' ')
            print ''
        sys.stdout.flush()
        """

        """
        print "wall height:", state.getWalls().height
        print "wall width:", state.getWalls().width
        print "wall count", state.getWalls().count()
        print "wall asList", len(state.getWalls().asList())
        print "wall asList, no key", len(state.getWalls().asList(False))
        print "wall data:", state.getWalls().data

        print "------------------------------------"
        print self.lastMyFoods.asList()
        print "------------------------------------"
        print self.lastTheirFoods
        print "------------------------------------"

        for x in range(1, self.width - 1):
            for y in range(1, self.height - 1):
                print (x,y) , ":", self.getLegalNeighbors((x, y))
        """


    # return next state based on action, can be used to obtain next own position
    def getNextState(self, state, action):
        successor = state.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()

        if pos != util.nearestPoint(pos):
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    # return capsules of both sides
    def getAllCapsules(self, state):
        return self.getCapsulesYouAreDefending(state) + self.getCapsules(state)

    # return non-walled neighbors of the pos
    # if the pos itself is wall, also returns []
    def getLegalNeighbors(self, pos):

        neighborList = []
        x, y = pos
        moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        if self.walls[x][y]:
            return []

        for move in moves:
            newx = x + move[0]
            newy = y + move[1]
            if newx > 0 and newx < self.width and newy > 0 and newy < self.height:
                if not self.walls[newx][newy]:
                    neighborList.append((newx, newy))

        return neighborList

    # get distance quickly, generally pass food as pos1
    def quickGetDistance(self, pos1, pos2):

        if pos1 == pos2:
            return 0

        key = (pos1, pos2)

        if key not in self.cachedDistance:
            self.cachedDistance[key] = self.getMazeDistance(pos1, pos2)

        return self.cachedDistance[key]

    # compute dead ends, call this only when initializing or capsule updated
    def computeDeadEnds(self, state):

        # key : a pos that is in a dead end
        # value : the exit of the dead end
        self.deadEnds = {}

        # a list of positions that have exactly one neighbor
        odCorners = []

        for x in range(1, self.width - 1):
            for y in range(1, self.height - 1):
                pos = (x, y)
                if len(self.getLegalNeighbors(pos)) == 1:
                    odCorners.append(pos)

        for corner in odCorners:
            neighbor = self.getLegalNeighbors(corner)
            next = corner
            positions = []

            isCapsuleContained = False

            while len(neighbor) > 0 and len(neighbor) <= 2 and next not in positions:

                positions.append(next)

                if next in self.getAllCapsules(state):
                    isCapsuleContained = True
                    break

                for nei in neighbor:
                    if nei not in positions:
                        next = nei

                neighbor = self.getLegalNeighbors(next)

            # next is exit and no capsule in the dead end
            if len(neighbor) >= 3 and not isCapsuleContained:
                for pos in positions:
                    self.deadEnds[pos] = next

    # return position of foods that were in old but not in new
    def compareFoods(self, old, new):

        foodList = []

        for x in range(old.width):
            for y in range(old.height):
                if old[x][y] and not new[x][y]:
                    foodList.append((x, y))

        return foodList

    # return invaders' positions, including those who are invisible but detected by food change
    def getInvadingEnemyPositions(self, state):

        nowMyFoods = self.getFoodYouAreDefending(state)

        enemyPositions = self.compareFoods(self.lastMyFoods, nowMyFoods)

        for index in self.getOpponents(state):
            enemyState = state.getAgentState(index)
            pos = enemyState.getPosition()
            if enemyState.isPacman and pos is not None and pos not in enemyPositions:
                enemyPositions.append(pos)

        return enemyPositions

    # return positions of visible unscared ghost enemies
    def getUnscaredGhostEnemyPositions(self, state):

        enemyPositions = []

        for index in self.getOpponents(state):
            enemyState = state.getAgentState(index)
            pos = enemyState.getPosition()
            if not enemyState.isPacman and pos is not None and enemyState.scaredTimer <=0:
                enemyPositions.append(pos)

    # return an action list moving toward or away from the target
    def movingRelativeToTarget(self, state, target, isCloser = True):

        myPos = state.getAgentState(self.index).getPosition()

            

        return Directions.STOP

    # return a defensive action if possible
    def defend(self, state):


        # TODO
        # perform required action sequence

        actions = state.getLegalActions(self.index)

        #for action in actions:
            #print self.getNextState(state, action).getAgentState(self.index).getPosition()

        time.sleep(1)

        return None

    # return a offensive action if possible
    def offend(self, state):

        return None

    # return a more offensive action, likely to get eaten
    def offendAtRisk(self, state):

        return None


    def chooseAction(self, state):

        scaredTimer = state.getAgentState(self.index).scaredTimer

        action = None

        # temporarily become offensive is scared
        if scaredTimer > 0:
            action = self.offend(state)
            if action is None:
                action = self.move



        action = self.defend(state)

        # update lastMyFoods
        self.lastMyFoods = self.getFoodYouAreDefending(state)

        return action

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
    if pos != util.nearestPoint(pos):
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
