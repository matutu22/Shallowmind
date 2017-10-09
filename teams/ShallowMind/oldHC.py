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

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'DumbOffensiveAgent', second = 'DumbDefensiveAgent'):
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


# a Dumb Quirky Naive agent
class DumbDefensiveAgent(CaptureAgent):

    def registerInitialState(self, state):

        CaptureAgent.registerInitialState(self, state)

        self.walls = state.getWalls().data
        self.height = state.getWalls().height
        self.width = state.getWalls().width
        self.homes = self.getHomeInMid()
        self.startPos = state.getAgentState(self.index).getPosition()
        self.enemyStartPos = [state.getAgentState(i).getPosition() for i in self.getOpponents(state)][0]

        self.cachedDistance = {}

        self.isChasingDD = False
        self.chasingTarget = None
        self.chasingDest = None

        # last status, make sure update this in chooseAction
        # do not get these status based on self.getPreviousObservation
        self.lastMyFoods = self.getFoodYouAreDefending(state)
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

    # among a bunch of targets' positions, which one is the closest to source
    # return position, or None if failed
    def quickFindMinDisInList(self, source, targets):

        if source is None or targets is None or len(targets) <= 0:
            return None

        minDis = 9999999
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
    def getUnscaredGhostEnemyPositions(self, state):

        enemyPositions = []

        for index in self.getOpponents(state):
            enemyState = state.getAgentState(index)
            pos = enemyState.getPosition()
            if not enemyState.isPacman and pos is not None and enemyState.scaredTimer <=0:
                enemyPositions.append(pos)

        return enemyPositions

    # return a action list moving toward or away from the target
    def movingRelativeToTarget(self, state, target, isCloser = True):

        if target is None:
            return {}

        actions = state.getLegalActions(self.index)

        results = {}

        for action in actions:
            successor = self.getNextState(state, action)
            distance = self.quickGetDistance(target, successor.getAgentPosition(self.index))
            results[action] = distance

        if len(results) <= 0:
            return {}

        if isCloser:
            bestValue = min(results.itervalues())
        else:
            bestValue = max(results.itervalues())

        bestActions = [key for key in results.keys() if results[key] == bestValue]

        if bestActions is None:
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
                return self.movingRelativeToTarget(state, self.chasingDest, True)
            # blocking
            else:
                # just stay if its in there
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

        closetEmemy = self.quickFindMinDisInList(myPos, enemies)

        if closetEmemy is None:

            closetEmemy = self.quickFindMinDisInList(myPos, self.lastInvadingEnemyPos)

        if closetEmemy is not None:
            return self.movingRelativeToTarget(state, closetEmemy, True)

        # no idea where are the enemies
        else:
            mostPossibleFood = self.quickFindMinDisInList(self.enemyStartPos, self.getFoodYouAreDefending(state).asList())
            mostPossibleCapsule = self.quickFindMinDisInList(self.enemyStartPos, self.getCapsulesYouAreDefending(state))

            possibleDisToFood = 999999
            possibleDisToCapsule = 999999

            if mostPossibleFood is not None:
                possibleDisToFood = self.quickGetDistance(self.enemyStartPos, mostPossibleFood)

            if mostPossibleCapsule is not None:
                # tend to defend capsule
                possibleDisToCapsule = self.quickGetDistance(self.enemyStartPos, mostPossibleCapsule) - 6

            if possibleDisToCapsule < possibleDisToFood:
                return self.movingRelativeToTarget(state, mostPossibleCapsule, True)
            else:
                if mostPossibleFood is not None:
                    return self.movingRelativeToTarget(state, mostPossibleFood, True)

        return None


    # find a shortest path to one of these food
    def goForThem(self, state, foods):

        myPos = state.getAgentState(self.index).getPosition()

        visibleGhosts = self.getUnscaredGhostEnemyPositions(state)

        bestFood = None
        bestFoodDis = 999999

        if len(foods) > 0:
            for food in foods:
                minGhostToFood = self.quickFindMinDisInList(food, visibleGhosts)
                if minGhostToFood is None or self.quickGetDistance(food, myPos) < self.quickGetDistance(food, minGhostToFood):
                    if self.quickGetDistance(food, myPos) < bestFoodDis:
                        bestFoodDis = self.quickGetDistance(food, myPos)
                        bestFood = food

        capsules = self.getCapsules(state)

        bestCapsule = None
        bestCapsuleDis = 999999

        if len(capsules) > 0:
            for capsule in capsules:
                minGhostToCapsule = self.quickFindMinDisInList(capsule, visibleGhosts)
                if minGhostToCapsule is None or self.quickGetDistance(capsule, myPos) < self.quickGetDistance(capsule, minGhostToCapsule):
                    if self.quickGetDistance(capsule, myPos) < bestCapsuleDis:
                        bestCapsuleDis = self.quickGetDistance(capsule, myPos)
                        bestCapsule = capsule

        if bestCapsuleDis - 5 < bestFoodDis:
            if bestCapsule is not None:
                return self.movingRelativeToTarget(state, bestCapsule)

        else:
            if bestFood is not None:
                return self.movingRelativeToTarget(state, bestFood)

        return None

    # return a offensive action if possible
    def offend(self, state):

        foods = []

        # relatively safe foods
        for food in self.getFood(state).asList():
            if food not in self.deadEnds:
                foods.append(food)

        return self.goForThem(state, foods)


    # TODO need to be modified
    # return a more offensive action, likely to get eaten
    def offendAtRisk(self, state):

        return self.goForThem(state,self.getFood(state).asList())

    # return an action to go home if safe
    def moveBackHome(self, state):

        myPos = state.getAgentState(self.index).getPosition()

        safeHouses = []

        visibleGhosts = self.getUnscaredGhostEnemyPositions(state)

        if len(visibleGhosts) <=0:
            safeHouses = self.homes
        else:
            for home in self.homes:
                if self.quickGetDistance(home, myPos) < self.quickGetDistance(home, self.quickFindMinDisInList(home, visibleGhosts)):
                    safeHouses.append(home)

        if len(safeHouses) > 0:
            dest = self.quickFindMinDisInList(myPos, safeHouses)
            return self.movingRelativeToTarget(state, dest)

        return None

    # randomly move
    def randomMove(self, state):

        return random.choice(state.getLegalActions(self.index))


    def chooseAction(self, state):

        ownState = state.getAgentState(self.index)

        scaredTimer = ownState.scaredTimer

        action = None

        # temporarily become offensive is scared
        if scaredTimer > 0:
            self.clearDDChasingBuff()
            action = self.offend(state)
            if action is None and ownState.numCarrying > 0:
                action  = self.moveBackHome(state)

        # no longer scared, but still on enemy's side
        elif ownState.isPacman:
            self.clearDDChasingBuff()
            action  = self.moveBackHome(state)

        else:

            action = self.chaseDeadEnd(state)

            if action is None:
                action = self.defend(state)

        if action is None:
            action = self.randomMove(state)

        # update last Status
        self.lastMyFoods = self.getFoodYouAreDefending(state)
        self.lastInvadingEnemyPos = self.getInvadingEnemyPositions(state)


        return action


# another Dumb Quirky Naive agent for offense
class DumbOffensiveAgent(DumbDefensiveAgent):

    def chooseAction(self, state):

        ownState = state.getAgentState(self.index)

        action = None

        # TODO check the distance between home and nearest food
        if ownState.numCarrying > 8:
            action = self.moveBackHome(state)

        if action is not None:
            return action

        action = self.offend(state)
        if action is None and ownState.numCarrying > 0:
            action  = self.moveBackHome(state)

        if action is not None:
            return action

        if action is None:
            action = self.offendAtRisk(state)

        if not ownState.isPacman:
            self.defend(state)

        if action is None:
            action = self.randomMove(state)

        # update last Status
        self.lastMyFoods = self.getFoodYouAreDefending(state)
        self.lastInvadingEnemyPos = self.getInvadingEnemyPositions(state)

        return action
