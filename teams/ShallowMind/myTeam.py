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
from game import Directions, Actions
import game
import json
import ast
from collections import defaultdict

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'DumbDefensiveAgent', second = 'QOffensiveAgent'):
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

### IMPORTANT : Modify Parameters before submitting!!

# Q-Learning Agent
class QAgent():

    # alpha: learning rate
    # gamma: discount
    # epsilon: exploration rate
    def __init__(self, index, alpha, gamma, epsilon, path):

        print "Q-Learning init"

        # CaptureAgent.__init__(self, index)

        # Notice: QAgent is no longer a subclass of CaptureAgent

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.path =  "./teams/ShallowMind/"  + path
        self.lastAction = None
        self.lastFeature = None

        self.loadQ()

        # using load here when tranning
        # it is linear Agent when having loadW
        # !!!!!!!!!!!
        # COMMENT following 4 lines for contest

        """
        if hasattr(self, 'loadW'):
            self.loadW()
        else:
            self.loadQ()
        """

    def loadQ(self):

        self.q = util.Counter()

        with open(self.path) as f:
            tmp = json.load(f)
            for k,v in tmp.iteritems():
                self.q[ast.literal_eval(k)] = v

    def saveQ(self):

        print "Saving Q to " + self.path + "..."

        with open(self.path, 'w') as f:
            json.dump({str(k): v for k, v in self.q.iteritems()}, f)

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

    # return best (features, action)
    def determineAction(self, state):

        qKeys = self.getQKeys(state)

        # epsilon-greedy explore
        if (util.flipCoin(self.epsilon)):
            return random.choice(qKeys)

        maxQ = self.getQMax(qKeys)

        bestPairs = [key for key in qKeys if self.getQValue(key[0]) == maxQ]

        #print state.getAgentState(self.index).getPosition(), state.getLegalActions(self.index)
        #print [(key, self.getQValue(key[0])) for key in qKeys]

        return random.choice(bestPairs)

    def update(self, oldFeature, newState, reward):

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


MAX_DISTANCE = 999999

CAPSULE_BETTER_THAN_FOOD = 6

# when enemy's ghost timer <= SCARED_TIMER_BOTTOM, regard it as over
SCARED_TIMER_BOTTOM = 5

# enemy is visible only when the Manhattan distance <= SIGHT_RANGE
SIGHT_RANGE = 5

# maximum steps for a agent
MAX_STEP = 300

# last moment
LAST_MOMENT = 50

# max unsafe food carrying
MAX_UNSAFE_FOOD_CARRYING = 3

# if 2 agents' distance <= MAX_HIT_DISTANCE, they might hit in the next step
MAX_HIT_DISTANCE = 2

# sometimes this dumb agent goes insane
INSANE_PROBABILITY = 0

# CHANGE STATE WHEN SCORE >= RestFood * SCORE_RATIO
SCORE_RATIO = 1.5

# be extremely aggressive if self.powerTimer >= SAFE_POWER_TIME
SAFE_POWER_TIME = 20


# a Dumb Quirky Naive defensive agent
class DumbDefensiveAgent(CaptureAgent):

    def registerInitialState(self, state):

        CaptureAgent.registerInitialState(self, state)

        self.walls = state.getWalls().data
        self.height = state.getWalls().height
        self.width = state.getWalls().width
        self.homes = self.getHomeInMid()
        self.startPos = state.getAgentState(self.index).getPosition()
        self.enemyStartPos = [state.getAgentState(i).getPosition() for i in self.getOpponents(state)][0]
        self.step = 0
        self.powerTimer = 0

        self.cachedDistance = {}

        self.isChasingDD = False
        self.chasingTarget = None
        self.chasingDest = None

        # only used for offensive agent
        self.movingSequence = []

        # last status, make sure update this in chooseAction
        # do not get these status based on self.getPreviousObservation
        self.lastMyFoods = self.getFoodYouAreDefending(state)
        self.lastInvadingEnemyPos = []
        self.lastAllCapules = self.getAllCapsules(state)
        self.lastTheirCapsules = self.getCapsules(state)
        self.lastPosSeq = []

        self.computeDeadEnds(state)
        self.aps = [realAP for realAP in self.getArticulationNode(state) if realAP not in self.deadEnds]

        # IMPORTANT!!!! ESPECIALLY FOR SUBCLASS QOffensiveAgent
        random.seed(time.time())

        """
        import sys
        #print the map
        #for y in reversed(range(self.height)):
            #for x in range(self.width):
                if self.walls[x][y]:
                    sys.stdout.write('#')
                elif (x,y) in self.aps: #and (x,y) not in self.deadEnds:
                    sys.stdout.write('x')
                if (x,y) in self.deadEnds:
                    sys.stdout.write('@')
                elif not self.walls[x][y] and not (x,y) in self.aps:
                    sys.stdout.write(' ')
            print ''
        sys.stdout.flush()
        time.sleep(500)
        #for x in range(1, self.width - 1):
            #for y in range(1, self.height - 1):
                #print (x,y) , ":", self.getLegalNeighbors((x, y))
        """


    # return capsules of both sides
    def getAllCapsules(self, state):
        return self.getCapsulesYouAreDefending(state) + self.getCapsules(state)

    def getHomeInMid(self):

        homes = []

        midX = self.width / 2

        if self.red:
            midX -= 1

        for y in range(1, self.height - 1):
            if not self.walls[midX][y]:
                homes.append((midX, y))

        return homes

    # return non-walled neighbors of the pos
    # if the pos itself is wall, also returns []
    def getLegalNeighbors(self, pos):

        neighborList = []
        x, y = pos
        x = int(x)
        y = int(y)
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

    # return the next position performing action at current position
    # this doesn't consider if the action is legal or not
    def getNextPos(self, currentPos, action):
        return Actions.getSuccessor(currentPos, action)

    # get distance quickly, generally pass food as pos1
    def quickGetDistance(self, pos1, pos2):

        if pos1 == pos2:
            return 0

        key = (pos1, pos2)

        if key not in self.cachedDistance:
            self.cachedDistance[key] = self.getMazeDistance(pos1, pos2)

        return self.cachedDistance[key]

    def getManhattanDistance(self, pos1, pos2):
        if pos1 is None or pos2 is None:
            return None

        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    # among a bunch of targets' positions, which one is the closest to source
    # return position, or None if failed
    def quickFindClosetPosInList(self, source, targets):

        if source is None or targets is None or len(targets) <= 0:
            return None

        minDis = MAX_DISTANCE
        minP = None

        for target in targets:
            if target is not None:
                if self.quickGetDistance(source, target) < minDis:
                    minP = target
                    minDis = self.quickGetDistance(source, target)

        return minP

    # compute dead ends, call this only when initializing or capsule updated
    def computeDeadEnds(self, state):

        # key : a pos that is in a dead end
        # value : the exit of the dead end
        self.deadEnds = {}

        # a list of positions that have exactly one neighbor
        odCorners = []

        # dynamically update the number of neighbor for this computation
        neighborList = {}

        for x in range(1, self.width - 1):
            for y in range(1, self.height - 1):
                pos = (x, y)
                neighborList[pos] = self.getLegalNeighbors(pos)
                if len(self.getLegalNeighbors(pos)) == 1:
                    odCorners.append(pos)

        for corner in odCorners:
            neighbor = neighborList[corner]
            next = corner
            positions = []

            isCapsuleContained = False

            while len(neighbor) == 1 and next not in positions:


                #print "next", next
                positions.append(next)

                if next in self.getAllCapsules(state):
                    isCapsuleContained = True
                    break

                for nei in neighbor:
                    if nei not in positions:
                        next = nei

                neighborList[next].remove(positions[-1])

                neighbor = neighborList[next]
                #print "neighbor:", neighbor

            # next is exit and no capsule in the dead end
            if len(neighbor) > 1 and not isCapsuleContained:
                for pos in positions:
                    self.deadEnds[pos] = next

    def getArticulationNode(self, state):
        nodeList = []
        for x in range(state.data.layout.width):
          for y in range(state.data.layout.height):
            if not state.hasWall(x, y):
              nodeList.append((x,y))

        nodeList.sort()
        edgeList = []
        for node in nodeList:
          for node1 in nodeList:
            if abs(node[0] - node1[0]) == 1 and node[1] - node1[1] ==0:
              edgeList.append((node, node1))
            elif node[0] - node1[0] == 0 and abs(node[1] - node1[1]) == 1:
              edgeList.append((node, node1))

        g = Graph(len(nodeList))
        nodeDict = {}
        for index, node in enumerate(nodeList):
            nodeDict[node] = index

        for edge in sorted(edgeList):
        #print g.graph
            g.addEdge(nodeDict[edge[0]], nodeDict[edge[1]])
        a = g.AP()
        articulationNodes = []
        for node, index in nodeDict.iteritems():
            if index in a:
                articulationNodes.append(node)
        return articulationNodes

    # return position of foods that were in old but not in new
    def compareFoods(self, old, new):

        foodList = []

        for x in range(old.width):
            for y in range(old.height):
                if old[x][y] and not new[x][y]:
                    foodList.append((x, y))

        return foodList

    # return enemy's index at certain position, none if failed
    def getEnemyIndexBasedOnPos(self, state, position):

        for index in self.getOpponents(state):
            enemyState = state.getAgentState(index)
            pos = enemyState.getPosition()
            if pos == position:
                return index

        return None

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
    # NOTICE: this will ignore the ghosts that are visible to my teammates but too far away from myself
    def getUnscaredGhostEnemyPositions(self, state):

        enemyPositions = []

        myPos = state.getAgentState(self.index).getPosition()

        for index in self.getOpponents(state):
            enemyState = state.getAgentState(index)
            pos = enemyState.getPosition()
            if (not enemyState.isPacman and pos is not None
            and enemyState.scaredTimer <= SCARED_TIMER_BOTTOM
            and self.getManhattanDistance(myPos, pos) <= SIGHT_RANGE + MAX_HIT_DISTANCE):
                enemyPositions.append(pos)

        return enemyPositions

    # A* based on Manhattan, find a path
    # if walkInOwnSide is True, the path wont contain nodes in enemy's side
    def aStarSearchMove(self, start, goal, walkInOwnSide = True):

        visited = []

        queue = util.PriorityQueue()

        queue.push((start, []), 0)

        while not queue.isEmpty():
            pos, path = queue.pop()

            if pos == goal:
                return path

            if not pos in visited:
                neighbors = self.getLegalNeighbors(pos)
                for nei in neighbors:
                    if not nei in visited:
                        if not walkInOwnSide or not self.isPacmanWhenInPos(nei):
                            newPath = path + [nei]
                            queue.push((nei, newPath), self.getManhattanDistance(nei, goal))

            visited.append(pos)

        return []


    # return a action moving toward  the target
    # can only be called when defending
    # when there are multiple choices, randomly choose one
    def defensiveMovingToTarget(self, state, target):

        if target is None:
            return None

        actions = state.getLegalActions(self.index)

        results = {}

        for action in actions:
            nextPos = self.getNextPos(state.getAgentState(self.index).getPosition(), action)
            # should not move to enemy's side
            if not self.isPacmanWhenInPos(nextPos):
                distance = self.quickGetDistance(target, nextPos)
                results[action] = distance

        if len(results) <= 0:
            return None

        bestValue = min(results.itervalues())

        bestActions = [key for key in results.keys() if results[key] == bestValue]

        if len(bestActions) <= 0:
            return None

        # TODO predict enemy's move
        return random.choice(bestActions)

    # return a action moving toward  the target
    # called when offending
    # when there are multiple choices, randomly choose one
    def offensiveMovingToTarget(self, state, target):

        if target is None:
            return None

        actions = state.getLegalActions(self.index)

        results = {}

        enemies = self.getUnscaredGhostEnemyPositions(state)

        minDis = MAX_DISTANCE
        minEnemyDis = MAX_DISTANCE

        for action in actions:
            nextPos = self.getNextPos(state.getAgentState(self.index).getPosition(), action)
            distance = self.quickGetDistance(target, nextPos)
            if distance < minDis:

                # found better pos, drop old values
                results.clear()
                results[action] = distance
                minDis = distance

                closetEnemy =  self.quickFindClosetPosInList(nextPos, enemies)
                if closetEnemy is None:
                    minEnemyDis = MAX_DISTANCE
                else:
                    minEnemyDis = self.quickGetDistance(nextPos, closetEnemy)

            elif distance == minDis:
                closetEnemy = self.quickFindClosetPosInList(nextPos, enemies)
                if closetEnemy is None:
                    results[action] = distance
                # the farther away from enemies, the better the pos is
                elif self.quickGetDistance(nextPos, closetEnemy) > minEnemyDis:
                    results.clear()
                    results[action] = distance
                    minDis = distance
                    minEnemyDis = self.quickGetDistance(nextPos, closetEnemy)


            #results[action] = distance

        if len(results) <= 0:
            return None

        bestActions = [key for key in results.keys()]

        if len(bestActions) <= 0:
            return None

        return random.choice(bestActions)

    def clearDDChasingBuff(self):
        self.isChasingDD = False
        self.chasingTarget = None
        self.chasingDest = None

    # a special defensive move
    # at most one agent should do this
    def chaseDeadEnd(self, state):

        ownState = state.getAgentState(self.index)

        # currently not not chasing, find if there is any chasable enemy
        if not self.isChasingDD or self.chasingTarget is None:

            enemies = self.getInvadingEnemyPositions(state)

            if len(enemies) > 0:
                for enemyPos in enemies:
                    if enemyPos in self.deadEnds:
                        dest = self.deadEnds[enemyPos]
                        # should chase it
                        if self.quickGetDistance(dest, ownState.getPosition()) <= self.quickGetDistance(dest, enemyPos):
                            self.isChasingDD = True
                            self.chasingTarget = self.getEnemyIndexBasedOnPos(state, enemyPos)
                            self.chasingDest = dest
                            if self.chasingTarget is not None:
                                break

        # check again, might start chasing
        if self.isChasingDD:

            lastState = self.getPreviousObservation()

            # illegal state
            if lastState is None:
                return None

            # can actually locate the target index (i.e. not through food)
            # check if the target dies
            if self.chasingTarget is not None:
                lastChasingTargetPos = lastState.getAgentState(self.chasingTarget).getPosition()
                nowChasingTargetPos = state.getAgentState(self.chasingTarget).getPosition()
                lastMyPos = lastState.getAgentState(self.index).getPosition()

                # chasing the target died
                if (lastChasingTargetPos is not None
                and self.quickGetDistance(lastChasingTargetPos, lastMyPos) <= 3
                and nowChasingTargetPos is None):
                        self.clearDDChasingBuff()
                        return None

            # just chasing
            if ownState.getPosition != self.chasingDest:
                return self.defensiveMovingToTarget(state, self.chasingDest)
            # blocking
            else:
                #just stay if its in there
                if self.chasingTarget is not None or util.flipcoin(0.7):
                    return Directions.STOP
                # cannot know if its in there, so might just leave
                else:
                    self.clearDDChasingBuff()
                    return None


    # return a defensive action if possible
    def defend(self, state):

        # TODO
        # perform required action sequence

        myPos = state.getAgentState(self.index).getPosition()

        actions = state.getLegalActions(self.index)

        enemies = self.getInvadingEnemyPositions(state)


        closetEmemy = self.quickFindClosetPosInList(myPos, enemies)

        # no visible enemies, use their last positions
        if closetEmemy is None:
            closetEmemy = self.quickFindClosetPosInList(myPos, self.lastInvadingEnemyPos)

        # chase enemy
        if closetEmemy is not None:
                return self.defensiveMovingToTarget(state, closetEmemy)

        # no idea where are the enemies
        else:
            mostPossibleFood = self.quickFindClosetPosInList(self.enemyStartPos, self.getFoodYouAreDefending(state).asList())
            mostPossibleCapsule = self.quickFindClosetPosInList(self.enemyStartPos, self.getCapsulesYouAreDefending(state))

            possibleDisToFood = MAX_DISTANCE
            possibleDisToCapsule = MAX_DISTANCE

            if mostPossibleFood is not None:
                possibleDisToFood = self.quickGetDistance(self.enemyStartPos, mostPossibleFood)

            if mostPossibleCapsule is not None:
                # should tend to defend capsule,  minus CAPSULE_BETTER_THAN_FOOD
                possibleDisToCapsule = self.quickGetDistance(self.enemyStartPos, mostPossibleCapsule) - CAPSULE_BETTER_THAN_FOOD

            chosenDest = None
            if possibleDisToCapsule < possibleDisToFood:
                chosenDest = mostPossibleCapsule
            elif mostPossibleFood is not None:
                chosenDest = mostPossibleFood

            if chosenDest is not None:

                if self.red:
                    nearXHome = int(self.homes[0][0] - 1)
                else:
                    nearXHome = int(self.homes[0][0] + 1)

                posInMid = None
                minDis = MAX_DISTANCE

                for home in self.homes:
                    # choose a place which is not behind a wall
                    if not self.walls[nearXHome][int(home[1])]:
                        if self.quickGetDistance(home, chosenDest) < minDis:
                            posInMid = home
                            minDis = self.quickGetDistance(home, chosenDest)

                # to the pos in the mid of map
                if posInMid is not None and self.getManhattanDistance(chosenDest, posInMid) <= SIGHT_RANGE:
                    return self.defensiveMovingToTarget(state, posInMid)
                # or directly to that food/capsule
                else:
                    return self.defensiveMovingToTarget(state, chosenDest)

        return None

    def numFoodMyTeamCarrying(self, state):

        teamIndices = self.getTeam(state)

        return sum(state.getAgentState(index).numCarrying for index in teamIndices)

    # find a safe and closest pos in intendedList
    # return the (bestPos, bestDis), (None, MAX_DISTANCE) if all is unsafe
    def findBestSafeChoiceAmong(self, myPos, intendedList, ghostList):

        bestPos = None
        bestPosDis = MAX_DISTANCE

        if len(intendedList) > 0:
            for pos in intendedList:
                minGhostToFood = self.quickFindClosetPosInList(pos, ghostList)
                if minGhostToFood is None or self.quickGetDistance(pos, myPos) < self.quickGetDistance(pos, minGhostToFood):
                    if self.quickGetDistance(pos, myPos) < bestPosDis:
                        bestPosDis = self.quickGetDistance(pos, myPos)
                        bestPos = pos


        return (bestPos, bestPosDis)

    # find a probably unsafe but closest pos in intendedList
    # return the (bestPos, bestDis), (None, MAX_DISTANCE) if all is too unsafe
    def findBestUnsafeChoiceAmong(self, myPos, intendedList, ghostList):

        bestPos = None
        bestPosDis = MAX_DISTANCE

        if len(intendedList) > 0:
            for pos in intendedList:
                minGhostToMe = self.quickFindClosetPosInList(myPos, ghostList)
                minGhostToFood = self.quickFindClosetPosInList(pos, ghostList)
                if (minGhostToFood is None
                or self.quickGetDistance(pos, myPos) < self.quickGetDistance(pos, minGhostToFood)
                or (self.quickGetDistance(myPos, minGhostToMe) > MAX_HIT_DISTANCE * 2
                    and self.quickGetDistance(pos, minGhostToMe) > SIGHT_RANGE and minGhostToMe == minGhostToFood)):
                    if self.quickGetDistance(pos, myPos) < bestPosDis:
                        bestPosDis = self.quickGetDistance(pos, myPos)
                        bestPos = pos

        return (bestPos, bestPosDis)

    # if nextPos is in enemy's side
    # do not use isPacman in successor.getAgentState
    def isPacmanWhenInPos(self, nextPos):
        myX = nextPos[0]
        homeX = self.homes[0][0]
        if self.red:
            return myX > homeX
        else:
            return myX < homeX

    # the absolute x manhattan distance
    def xManhattanDisToHomeFrontier(self, pos):
        myX = pos[0]
        homeX = self.homes[0][0]
        return abs(myX - homeX)

    # go to the target, no matter if the agent gonna die
    def hitWhatever(self, state, target):

        actions = state.getLegalActions(self.index)
        minDisToExit = MAX_DISTANCE
        minAction = None
        myPos = state.getAgentState(self.index).getPosition()

        # just move there
        for action in actions:
            dis = self.quickGetDistance(target, self.getNextPos(myPos, action))
            if dis < minDisToExit:
                minDisToExit = dis
                minAction = action

        return minAction


    # called in both sides ?
    def isCloserToValuableTarget(self, oldState, lastPos, nowPos):

        # compared to the foodlist of oldState
        foodlist = self.getFood(oldState).asList()
        capsuleList = self.getCapsules(oldState)

        if len(foodlist) > 0:
            nowMinFoodDistance = min([self.quickGetDistance(food, nowPos) for food in foodlist])
            lastMinFoodDistance = min([self.quickGetDistance(food, lastPos) for food in foodlist])
            if nowMinFoodDistance < lastMinFoodDistance:
                return True

        if len(capsuleList) > 0:
            nowDis = min([self.quickGetDistance(capsule, nowPos) for capsule in capsuleList])
            lastDis =  min([self.quickGetDistance(capsule, lastPos) for capsule in capsuleList])
            if nowDis < lastDis:
                return True

        return False

    # find a shortest path to capsule or parameter foods
    def goForThem(self, state, foods):

        myPos = state.getAgentState(self.index).getPosition()

        visibleGhosts = self.getUnscaredGhostEnemyPositions(state)

        bestFood = None
        bestFoodDis = MAX_DISTANCE

        bestFood, bestFoodDis = self.findBestSafeChoiceAmong(myPos, foods, visibleGhosts)

        capsules = self.getCapsules(state)

        bestCapsule, bestCapsuleDis =  self.findBestSafeChoiceAmong(myPos, capsules, visibleGhosts)

        # find probably unsafe choice
        if bestCapsule is None and bestFood is None:
            bestFood, bestFoodDis = self.findBestUnsafeChoiceAmong(myPos, foods, visibleGhosts)
            bestCapsule, bestCapsuleDis =  self.findBestUnsafeChoiceAmong(myPos, capsules, visibleGhosts)

        if len(capsules) > 0:
            for capsule in capsules:
                minGhostToCapsule = self.quickFindClosetPosInList(capsule, visibleGhosts)
                if minGhostToCapsule is None or self.quickGetDistance(capsule, myPos) < self.quickGetDistance(capsule, minGhostToCapsule):
                    if self.quickGetDistance(capsule, myPos) < bestCapsuleDis:
                        bestCapsuleDis = self.quickGetDistance(capsule, myPos)
                        bestCapsule = capsule

        # prefer capsule if neccessary
        if bestCapsuleDis - (CAPSULE_BETTER_THAN_FOOD - 1) < bestFoodDis:
            if bestCapsule is not None and self.powerTimer <= bestCapsuleDis + MAX_HIT_DISTANCE:
                return self.offensiveMovingToTarget(state, bestCapsule)

        else:
            if bestFood is not None:
                return self.offensiveMovingToTarget(state, bestFood)

        return None

    # return a offensive action if possible
    def offend(self, state):

        foods = []

        myPos = state.getAgentState(self.index).getPosition()
        closetEnemy = self.quickFindClosetPosInList(myPos, self.getUnscaredGhostEnemyPositions(state))
        minDisToEnemy = SIGHT_RANGE + 1

        if self.powerTimer > SAFE_POWER_TIME and closetEnemy is None:
            return self.goForThem(state, self.getFood(state).asList())



        # relatively safe foods
        for food in self.getFood(state).asList():
            if food not in self.deadEnds:
                foods.append(food)
            else:
                # enemy's distance between
                if closetEnemy is not None:
                    minDisToEnemy = self.quickGetDistance(self.deadEnds[food], closetEnemy)
                if (self.quickGetDistance(myPos, food) * 2 + self.quickGetDistance(myPos, self.deadEnds[food]) + 1 < minDisToEnemy
                    and self.quickGetDistance(food, self.deadEnds[food]) <= SIGHT_RANGE):
                    foods.append(food)

        return self.goForThem(state, foods)


    # TODO need to be modified
    # return a more offensive action, likely to get eaten
    def offendAtRisk(self, state):

        return self.goForThem(state,self.getFood(state).asList())

    # return an action to go home
    # if mightFail is False, this method will only return None if not 100% sure can safely reach home
    # if mightFail is True, return an action that might not actually reach home when no other choices left
    def moveBackHome(self, state, mightFail):

        myPos = state.getAgentState(self.index).getPosition()

        safeHouses = []

        visibleGhosts = self.getUnscaredGhostEnemyPositions(state)

        if len(visibleGhosts) <=0:
            safeHouses = self.homes
        else:
            for home in self.homes:
                if self.quickGetDistance(home, myPos) < self.quickGetDistance(home, self.quickFindClosetPosInList(home, visibleGhosts)):
                    safeHouses.append(home)

        if len(safeHouses) > 0:
            dest = self.quickFindClosetPosInList(myPos, safeHouses)
            return self.offensiveMovingToTarget(state, dest)

        # might not succeesful come home if going this way
        if mightFail:
            for home in self.homes:
                closetToHomeEnemy = self.quickFindClosetPosInList(home, visibleGhosts)
                disToEnemy = self.quickGetDistance(myPos, closetToHomeEnemy)
                if disToEnemy > self.quickGetDistance(home, myPos):
                    safeHouses.append(home)

            if len(safeHouses) > 0:
                dest = self.quickFindClosetPosInList(myPos, safeHouses)
                return self.offensiveMovingToTarget(state, dest)

        return None

    # random move
    def randomMove(self, state):

        return random.choice(state.getLegalActions(self.index))

    # randomly choose a safe move if possible
    # NOTICE: wont pick the safest choice
    def safeRandomMove(self, state):

        actions = []

        ghosts = self.getUnscaredGhostEnemyPositions(state)

        # no enemy around here
        if len(ghosts) <= 0:
            return self.randomMove(state)

        myPos = state.getAgentState(self.index).getPosition()

        for action in state.getLegalActions(self.index):
            nextPos = self.getNextPos(myPos, action)
            closetEmemy = self.quickFindClosetPosInList(nextPos, ghosts)
            if self.quickGetDistance(nextPos, closetEmemy) >= MAX_HIT_DISTANCE or not self.isPacmanWhenInPos(nextPos):
                actions.append(action)

        if len(actions) > 0:
            return random.choice(actions)

        return self.randomMove(state)

    def updateMyStatus(self, state):

        # capsule changed, recompute all the dead ends
        if len(self.getAllCapsules(state)) < len(self.lastAllCapules):
            self.computeDeadEnds(state)
            # eaten their capsule
            if len(self.getCapsules(state)) < len(self.lastTheirCapsules):
                self.powerTimer = 40

        # update last Status
        self.lastMyFoods = self.getFoodYouAreDefending(state)
        self.lastInvadingEnemyPos = self.getInvadingEnemyPositions(state)
        self.lastAllCapules = self.getAllCapsules(state)
        self.lastTheirCapsules = self.getCapsules(state)
        self.lastPosSeq.append(state.getAgentState(self.index).getPosition())
        self.step += 1
        if self.powerTimer > 0:
            self.powerTimer -= 1

    # for each step, only one of defensiveAction and offensiveAction should be called
    # when this method is called, make sure that scaredTimer <= 0
    def defensiveAction(self, state):

        ownState = state.getAgentState(self.index)

        action = None

        # no longer scared, but still on enemy's side
        if ownState.isPacman:
            self.clearDDChasingBuff()
            action  = self.moveBackHome(state, True)
            if action is None and ownState.getPosition() not in self.deadEnds:
                action = self.safeRandomMove(state)
            # get blocked
            elif action is None:
                self.hitWhatever(state, self.deadEnds[ownState.getPosition()])

        else:
            action = self.chaseDeadEnd(state)
            if action is None:
                action = self.defend(state)

        if action is None:
            action = self.randomMove(state)

        return action

    def offensiveAction(self, state):

        ownState = state.getAgentState(self.index)

        self.clearDDChasingBuff()

        action = None

        minDisToHome = self.quickGetDistance(ownState.getPosition(), self.quickFindClosetPosInList(ownState.getPosition(), self.homes))

        winIfComeBack = self.getScore(state) + self.numFoodMyTeamCarrying(state) > 0

        # check the distance between home and nearest food and time left
        if ownState.numCarrying > 0 and MAX_STEP - self.step < 2.5 * minDisToHome and winIfComeBack:
            action = self.moveBackHome(state, True)

        # only 2 dots left
        if action is None and len(self.getFood(state).asList()) <= 2 and ownState.numCarrying > 0:
            action = self.moveBackHome(state, True)

        if action is None:
            action = self.offend(state)

        # go home if not too hard
        if action is None and ownState.numCarrying > MAX_UNSAFE_FOOD_CARRYING and winIfComeBack:
            action  = self.moveBackHome(state, False)

        # go home if not too hard
        if action is None and ownState.numCarrying > 0 and self.getScore(state) <= 0 and winIfComeBack:
            action  = self.moveBackHome(state, False)

        if action is None:
            action = self.offendAtRisk(state)

        if action is None and ownState.numCarrying > 0:
            action  = self.moveBackHome(state, True)

        if action is None:
                action = self.safeRandomMove(state)

        return action

    # called by the system
    def chooseAction(self, state):

        ownState = state.getAgentState(self.index)

        numMyFoodLeft = len(self.getFoodYouAreDefending(state).asList())

        minDisToHome = self.quickGetDistance(ownState.getPosition(), self.quickFindClosetPosInList(ownState.getPosition(), self.homes))

        score = self.getScore(state)

        action = None

        scaredTimer = ownState.scaredTimer

        # temporarily become offensive if scared
        if scaredTimer > 0:
            self.clearDDChasingBuff()

            #go home as the timer gets closer to 0
            if scaredTimer <= minDisToHome and ownState.isPacman:
                action  = self.moveBackHome(state, True)

            # no need to go home, or failed to find a safe way
            if action is None:
                action = self.offensiveAction(state)

            # no way, but not in dead ends neither
            if action is None and ownState.getPosition() not in self.deadEnds:
                action = self.safeRandomMove(state)
            # blocked
            elif action is None:
                action = self.hitWhatever(state, self.deadEnds[ownState.getPosition()])

        # lost too much food, no longer should defend
        if score <= (-1) * numMyFoodLeft * SCORE_RATIO:
            self.clearDDChasingBuff()
            action = self.offensiveAction(state)

        if action is None:
        # last moment!
            if MAX_STEP - self.step <= 15 * (abs(score) - self.numFoodMyTeamCarrying(state)) + minDisToHome and score < 0:
                self.clearDDChasingBuff()
                action = self.offensiveAction(state)

        if action is None:
            action = self.defensiveAction(state)

        self.updateMyStatus(state)

        return action

# the max food carrying recorded in feature,  i.e. any > 8 is regarded as 8
MAX_FOOD_CARRYING = 8

# absolute manhattan distance to home, when self is ghost, always set as 0 to cut space
MAX_X_MDIS_TO_HOME = 10

# the max depth of death end, any deeper length will be regarded as this
MAX_DEATH_END_DEPTH = 4


# Q-Learning parameter
ALPHA = 0.2 #---------------------------- for contest!!
GAMMA = 0.85
EPSILON = 0.005  # for contest
QTABLE = "offense.json"

BRING_FOOD_REWARD_BASE = 10
BRING_FOOD_REWARD_RATIO = 20

GET_EATEN_REWARD_BASE = -100
GET_EATEN_REWARD_RATIO = -10

EAT_FOOD_REWARD = 10

EAT_CAPSULE_REWARD = 200

CLOSER_TO_FOOD_OR_CAPSULE_IN_ENEMY_SIDE = 0

IMPROPER_WALK_IN_ENEMY_SIDE = -0.2

WALKING_IN_HOME_TOWARD_FOOD_REWARD = -1

WALKING_IN_HOME_BACKWARD_REWARD = -3



class QOffensiveAgent(QAgent, DumbDefensiveAgent):

    def __init__(self, index, alpha = ALPHA, gamma = GAMMA, epsilon = EPSILON, path = QTABLE):

        CaptureAgent.__init__(self, index)
        QAgent.__init__(self, index, alpha, gamma, epsilon, path)

        self.isQOffesnvie = True


    # find best target
    def findTarget(self, state, myPos, ghostList):

        allFoods = self.getFood(state).asList()

        firstClassFoods = []

        if len(allFoods) > 0:
            for food in allFoods:
                if food not in self.deadEnds:
                    firstClassFoods.append(food)

        safeFood, bestFoodDis = self.findBestSafeChoiceAmong(myPos, firstClassFoods, ghostList)

        safeCapsule, bestCapsuleDis = self.findBestSafeChoiceAmong(myPos, self.getCapsules(state), ghostList)

        # no choices
        if safeFood is None and safeCapsule is None:
            #print " going to find unsafe choice"
            safeFood, bestFoodDis = self.findBestUnsafeChoiceAmong(myPos, firstClassFoods, ghostList)
            safeCapsule, bestCapsuleDis = self.findBestUnsafeChoiceAmong(myPos, self.getCapsules(state), ghostList)

        # still no choices
        if safeFood is None and safeCapsule is None:
            #print " going to find safe choice in allFoods"
            safeFood, bestFoodDis = self.findBestSafeChoiceAmong(myPos, allFoods, ghostList)

            # oh no
            if safeFood is None:
                #print " going to find unsafe choice in allFoods"
                safeFood, bestFoodDis = self.findBestUnsafeChoiceAmong(myPos, allFoods, ghostList)

        if safeFood is None:
            return safeCapsule

        if safeCapsule is None:
            return safeFood

        # neither is none
        if bestCapsuleDis - CAPSULE_BETTER_THAN_FOOD < bestFoodDis:
            return safeCapsule
        else:
            return safeFood

        return None

    # stuck in loop
    # should only be called by the offensive agent!
    def loopBreaker(self, state):

        isInLoop = False
        seqLen = 7
        numPos = 2
        lastset = set([])
        myPos = state.getAgentState(self.index).getPosition()
        enemies = self.getUnscaredGhostEnemyPositions(state)

        while len(self.lastPosSeq) > seqLen and not isInLoop and seqLen <= 25:
            lastSet = set(self.lastPosSeq[-seqLen:])
            if len(lastSet) <= numPos and myPos in lastSet:
                isInLoop = True
            else:
                seqLen += 3
                numPos += 1

        if not isInLoop:
            return None

        #in loop

        # in dead ends in enemy's side
        if myPos in self.deadEnds and state.getAgentState(self.index).isPacman:
            # exit
            # self.hitWhatever(state, self.deadEnds[myPos])
            self.movingSequence = self.aStarSearchMove(myPos, self.deadEnds[myPos], False)
            return None

        # stuck in own side:
        if not state.getAgentState(self.index).isPacman:
            target = myPos

            # stuck in own side, but cannot find the enemy (it might because the agent happens to run away from the blocking enemy)
            if len(enemies) <= 0:
                # choose a target at the homeEdge which is farthest
                for home in self.homes:
                    if self.getManhattanDistance(home, myPos) > self.getManhattanDistance(target, myPos):
                        target = home
            # find a new pos that is farthest away from the enemy (Manhattan is good enough)
            else:
                minDis = MAX_DISTANCE
                for home in self.homes:
                    closetEnemy = self.quickFindClosetPosInList(home, enemies)
                    disToClsoetEnemy = self.quickGetDistance(home, closetEnemy)
                    realDisToMe = len(self.aStarSearchMove(myPos, home, True))
                    # close a pos that is close to mypos but far from enemy
                    if realDisToMe < minDis and disToClsoetEnemy > SIGHT_RANGE + 1:
                        minDis = realDisToMe
                        target = home

            # set moving sequence and let sequence handler handle this
            if target != myPos:
                self.movingSequence = self.aStarSearchMove(myPos, target, True)
                return None

        # stuck in somewhere else
        ghosts = self.getUnscaredGhostEnemyPositions(state)

        togos = []
        for nei in self.getLegalNeighbors(myPos):
            if nei not in ghosts and nei not in lastSet:
                togos.append(nei)

        # there is a safe place to go
        if len(togos) > 0:
            target = random.choice(togos)
            action = self.offensiveMovingToTarget(state, target)
            if action is not None:
                return action

        #TODO AP BLOCK DETECTION:

        return self.safeRandomMove(state)


    # a special sequence handler for the offensive agent
    def sequenceHandler(self, state):

        myPos = state.getAgentState(self.index).getPosition()

        if len(self.movingSequence) <= 0:
            return None

        actions = state.getLegalActions(self.index)

        for action in actions:
            nextPos = self.getNextPos(myPos, action)
            # find the correct next pos
            if nextPos == self.movingSequence[0]:
                self.movingSequence.pop(0)
                return action

        # current position does not match the sequence, might be caused by get eaten as a scared ghost
        # just clear the sequence
        self.movingSequence = []
        return None

    def getFeatures(self, state, action):

        ownState = state.getAgentState(self.index)

        myPos = ownState.getPosition()
        nextPos = self.getNextPos(myPos, action)

        allGhostPos = self.getUnscaredGhostEnemyPositions(state)

        minDis = SIGHT_RANGE + 1
        minManhattanDis = SIGHT_RANGE + 1
        isBeingChased = 0
        numCloseEnemys = 0

        if len(allGhostPos) > 0:
            for ghostPos in allGhostPos:

                    if self.quickGetDistance(nextPos, ghostPos) <= 1:
                        numCloseEnemys += 1

                    minDis = min(minDis, self.quickGetDistance(nextPos, ghostPos))
                    minManhattanDis = min(minManhattanDis, self.getManhattanDistance(nextPos, ghostPos))

                    index = self.getEnemyIndexBasedOnPos(state, ghostPos)
                    if index is not None:
                        possibleGhostPos = self.getNextPos(ghostPos, state.getAgentState(index).getDirection())
                        if not self.walls[int(possibleGhostPos[0])][int(possibleGhostPos[1])]:
                            if self.quickGetDistance(nextPos, possibleGhostPos) < self.quickGetDistance(myPos, possibleGhostPos):
                                isBeingChased = 1

        foods = self.getFood(state).asList()
        capsules = self.getCapsules(state)

        canEatFood = int(nextPos in foods)
        canEatCapsule = int(nextPos in capsules)

        # do not use getSuccessor to check if this agent is pacman
        # because getSuccessor might set your next pos to startPos !!
        isPacman = int(self.isPacmanWhenInPos(nextPos))

        nowDisToHomeFrontier = self.quickGetDistance(myPos, self.quickFindClosetPosInList(myPos, self.homes))
        nextDisToHomeFrontier = self.quickGetDistance(nextPos, self.quickFindClosetPosInList(nextPos, self.homes))

        # is moving to the mid of the map (our home edge)
        isMovingToHomeFrontier = int(nextDisToHomeFrontier < nowDisToHomeFrontier)

        intendedTarget = self.findTarget(state, myPos, allGhostPos)

        isMovingToTarget = 0

        if intendedTarget is not None:
            isMovingToTarget = int(self.quickGetDistance(nextPos, intendedTarget) < self.quickGetDistance(myPos, intendedTarget))

        closetFoodPos = self.quickFindClosetPosInList(myPos, foods)
        closetCapsulePos = self.quickFindClosetPosInList(myPos, capsules)

        isMovingToClosetFood = 0
        isMovingToClosetCapsule = 0

        if closetFoodPos is not None:
            isMovingToClosetFood = int(self.quickGetDistance(nextPos, closetFoodPos) < self.quickGetDistance(myPos, closetFoodPos))

        deathEndDepth = 0

        if nextPos in self.deadEnds:
            deathEndDepth = min(self.quickGetDistance(self.deadEnds[nextPos], nextPos), MAX_DEATH_END_DEPTH)

        numCarrying = min(ownState.numCarrying, MAX_FOOD_CARRYING)

        xMDisToHomeFrontier = 0

        # no need to use that value when walking in our side
        if isPacman > 0:
            xMDisToHomeFrontier = min(self.xManhattanDisToHomeFrontier(nextPos), MAX_X_MDIS_TO_HOME)

        # 6 * 6 * 3 * 2 * 2 * 2 * 2
        #  * 2 * 2 * 2 * 2
        # * 5 * 8 * (11/2)
        features = (minDis, minManhattanDis, numCloseEnemys, isBeingChased, canEatFood, canEatCapsule,
            isPacman, isMovingToTarget, isMovingToClosetFood, isMovingToHomeFrontier,
            deathEndDepth, numCarrying, xMDisToHomeFrontier)

        return features

    def determineReward(self, oldState, oldAction, newState):

        reward = 0

        nowPos = newState.getAgentPosition(self.index)
        lastPos =  oldState.getAgentPosition(self.index)

        foods = self.getFood(newState).asList()

        diff = (newState.getAgentState(self.index).numReturned -
            oldState.getAgentState(self.index).numReturned)

        # bring food back
        if (diff > 0):
            #print "bring food back!!!"
            reward = diff * BRING_FOOD_REWARD_RATIO + BRING_FOOD_REWARD_BASE
        else:
            # eat new food
            if newState.getAgentState(self.index).numCarrying > oldState.getAgentState(self.index).numCarrying:
                #print "eat new food"
                reward = EAT_FOOD_REWARD
            # get eaten by enemy as a pacman
            # currently we dont punish the offensive agent when it gets eaten as a scared ghost
            # cuz that would make it try to avoid enemy even in our side (which is wrong in most cases)
            elif nowPos == self.startPos and oldState.getAgentState(self.index).isPacman:
                #print "get eaten T.T"
                reward = GET_EATEN_REWARD_BASE + oldState.getAgentState(self.index).numCarrying * GET_EATEN_REWARD_RATIO
            else:
                # eat capsule
                if nowPos in self.getCapsules(oldState):
                    #print "eat capsule"
                    reward = EAT_CAPSULE_REWARD
                # walking in the enemy's side (i.e. offensive)
                elif newState.getAgentState(self.index).isPacman:

                    # no punish
                    if len(foods) <= 0:
                        return CLOSER_TO_FOOD_OR_CAPSULE_IN_ENEMY_SIDE

                    #print "walking in the enemy's side"
                    lastGhosts = self.getUnscaredGhostEnemyPositions(oldState)
                    nowGhosts = self.getUnscaredGhostEnemyPositions(newState)

                    reward = IMPROPER_WALK_IN_ENEMY_SIDE

                    #print nowPos
                    # closer to food or capsule
                    if self.isCloserToValuableTarget(oldState, lastPos, nowPos):

                        reward = CLOSER_TO_FOOD_OR_CAPSULE_IN_ENEMY_SIDE

                    # carring food
                    elif newState.getAgentState(self.index).numCarrying > 0:

                        nowDisToHomeFrontier = self.quickGetDistance(nowPos, self.quickFindClosetPosInList(nowPos, self.homes))
                        lastDisToHomeFrontier = self.quickGetDistance(lastPos, self.quickFindClosetPosInList(lastPos, self.homes))

                        # is moving to the mid of the map (our home edge)
                        isMovingToHomeFrontier = int(nowDisToHomeFrontier < lastDisToHomeFrontier)

                        if isMovingToHomeFrontier:
                            reward = CLOSER_TO_FOOD_OR_CAPSULE_IN_ENEMY_SIDE
                        else:
                            if len(nowGhosts) > 0 and len(lastGhosts) > 0:
                                lastMinPos = self.quickFindClosetPosInList(lastPos, lastGhosts)
                                nowMinPos = self.quickFindClosetPosInList(nowPos, nowGhosts)
                                # escaping
                                if (self.quickGetDistance(nowPos, nowMinPos) <= self.quickGetDistance(lastPos, lastMinPos)):
                                    reward = CLOSER_TO_FOOD_OR_CAPSULE_IN_ENEMY_SIDE
                                else:
                                    reward = IMPROPER_WALK_IN_ENEMY_SIDE * 3
                            elif len(nowGhosts) <= 0 and len(lastGhosts) > 0:
                                reward = CLOSER_TO_FOOD_OR_CAPSULE_IN_ENEMY_SIDE
                            else:
                                reward = IMPROPER_WALK_IN_ENEMY_SIDE * 2

                # waling in our side
                else:

                    if len(foods) <= 0:
                        return 0

                    #print "walking in our side"

                    # stop in our side is terrible
                    if (lastPos == nowPos):
                        reward = WALKING_IN_HOME_BACKWARD_REWARD * 1.2

                    if self.isCloserToValuableTarget(oldState, lastPos, nowPos):
                        reward = WALKING_IN_HOME_TOWARD_FOOD_REWARD
                    else:
                        reward = WALKING_IN_HOME_BACKWARD_REWARD

        return reward


    def chooseAction(self, state):

        feature = None
        action = None

        # check special sequence
        action = self.sequenceHandler(state)

        loopBreaker = None

        if action is None:
            loopBreaker = self.loopBreaker(state)
            # check again as loopBreaker may update this
            action = self.sequenceHandler(state)

        lastState = self.getPreviousObservation()

        """
        #self.isQOffesnvie = False

        #if self.isQOffesnvie and self.getScore(state) < 0 and self.step >= (MAX_STEP / 2):
            #self.isQOffesnvie = False

        # update Q, no matter if this round is using loopBreaker
        # but if used loopBreaker last time, lastFeature will be None, thus need to be recomputed
        if self.isQOffesnvie and lastState is not None and self.lastAction is not None:
            if self.lastFeature is not None:
                reward = self.determineReward(lastState, self.lastAction, state)
                self.update(self.lastFeature, state, reward)
            else:
                tmpFeature = self.getFeatures(lastState, self.lastAction)
                reward = self.determineReward(lastState, self.lastAction, state)
                self.update(tmpFeature, state, reward)
        """

        # not using loopBreaker or sequenceHandler
        if loopBreaker is None and action is None:
            #if self.isQOffesnvie:
                #feature, action = self.determineAction(state)
            #else:
                action = self.offensiveAction(state)
        # not using sequenceHandler, must be loopBreaker
        elif loopBreaker is not None:
            action = loopBreaker

        # this only update status of the dumb agent class
        self.updateMyStatus(state)

        self.lastAction = action
        self.lastFeature = feature

        """
        ###############################
        mmm = state.getAgentState(self.index).getPosition()
        nnn = self.getNextPos(mmm, action)

        wori = self.quickFindClosetPosInList(nnn, self.getUnscaredGhostEnemyPositions(state))

        if (wori is not None and self.quickGetDistance(wori, nnn) <= 1 and self.isPacmanWhenInPos(nnn)
            and mmm not in self.deadEnds and loopBreaker is None) :
            print "BUG"
            print "now at:", mmm, "moving to", nnn
            time.sleep(50)
        """

        if action is None:
            feature, action = self.determineAction(state)

        return action

from collections import defaultdict

#This class represents an undirected graph
#using adjacency list representation
class Graph:

    def __init__(self,vertices):
        self.V= vertices #No. of vertices
        self.graph = defaultdict(list) # default dictionary to store graph
        self.Time = 0

    # function to add an edge to graph
    def addEdge(self,u,v):
        self.graph[u].append(v)
        #self.graph[v].append(u)

    '''A recursive function that find articulation points
    using DFS traversal
    u --> The vertex to be visited next
    visited[] --> keeps tract of visited vertices
    disc[] --> Stores discovery times of visited vertices
    parent[] --> Stores parent vertices in DFS tree
    ap[] --> Store articulation points'''
    def APUtil(self,u, visited, ap, parent, low, disc):

        #Count of children in current node
        children =0

        # Mark the current node as visited and print it
        visited[u]= True

        # Initialize discovery time and low value
        disc[u] = self.Time
        low[u] = self.Time
        self.Time += 1

        #Recur for all the vertices adjacent to this vertex
        for v in self.graph[u]:
            # If v is not visited yet, then make it a child of u
            # in DFS tree and recur for it
            if visited[v] == False :
                parent[v] = u
                children += 1
                self.APUtil(v, visited, ap, parent, low, disc)

                # Check if the subtree rooted with v has a connection to
                # one of the ancestors of u
                low[u] = min(low[u], low[v])

                # u is an articulation point in following cases
                # (1) u is root of DFS tree and has two or more chilren.
                if parent[u] == -1 and children > 1:
                    ap[u] = True

                #(2) If u is not root and low value of one of its child is more
                # than discovery value of u.
                if parent[u] != -1 and low[v] >= disc[u]:
                    ap[u] = True

                # Update low value of u for parent function calls
            elif v != parent[u]:
                low[u] = min(low[u], disc[v])


    #The function to do DFS traversal. It uses recursive APUtil()
    def AP(self):

        # Mark all the vertices as not visited
        # and Initialize parent and visited,
        # and ap(articulation point) arrays
        visited = [False] * (self.V)
        disc = [float("Inf")] * (self.V)
        low = [float("Inf")] * (self.V)
        parent = [-1] * (self.V)
        ap = [False] * (self.V) #To store articulation points

        # Call the recursive helper function
        # to find articulation points
        # in DFS tree rooted with vertex 'i'
        for i in range(self.V):
            if visited[i] == False:
                self.APUtil(i, visited, ap, parent, low, disc)
        indexList = []
        for index, value in enumerate (ap):
            if value == True:
                indexList.append(index)

        return indexList
