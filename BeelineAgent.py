# BeeLineAgent.py

from enum import Enum
import random
# import networkx as nx
import numpy as np

class Action:
    WALK = 0
    TURNLEFT = 1
    TURNRIGHT = 2
    GRAB = 3
    SHOOT = 4
    CLIMB = 5


class Direction(Enum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

enc_orientation = [[1, 0, 0, 0],   #Right
                   [0, 1, 0, 0],    #Up
                   [0, 0, 1, 0],    # Left
                   [0, 0, 0, 1]]    #Down
enc_heardscream = [0, 1]
enc_glitter = [0, 1]
enc_hasgold = [0, 1]
enc_hasarrow = [0, 1]


class Agent:

    def __init__(self):
        self.visitedlocations = []
        self.breezelocations=[]
        self.stenchlocations = []
        self.heardscream = False
        self.hasarrow  = True
        self.hasgold = True
        self.glitter = False

    def processPercepts(self, WORLD_SIZE, percept):
        """ BeeLineAgent_Process: called with new percepts after each action to return the next action """

        percept_str = ""
        if percept['stench']:
            percept_str += "Stench=True,"
            self.stenchlocations.append(str(percept['x']) + ',' + str(percept['y'])) if str(percept['x']) + ',' + str(percept['y']) not in self.stenchlocations else None
        else:
            percept_str += "Stench=False,"
        if percept['breeze']:
            percept_str += "Breeze=True,"
            self.breezelocations.append(str(percept['x']) + ',' + str(percept['y'])) if str(percept['x']) + ',' + str(percept['y']) not in self.breezelocations else None
        else:
            percept_str += "Breeze=False,"
        if percept['glitter']:
            percept_str += "Glitter=True,"
            self.glitter = True
        else:
            percept_str += "Glitter=False,"
        if percept['bump']:
            percept_str += "Bump=True,"
        else:
            percept_str += "Bump=False,"
        if percept['scream']:
            percept_str += "Scream=True,"
            self.heardscream = True
        else:
            percept_str += "Scream=False,"
        if percept['arrow']:
            percept_str += "Has Arrow=True,"
            self.hasarrow = True
        else:
            percept_str += "Has Arrow=False,"
        if percept['gold']:
            percept_str += "Has Gold=True"
            self.hasgold = True
        else:
            percept_str += "Has Gold=False"

        self.visitedlocations.append(str(percept['x']) + ',' + str(percept['y'])) if str(percept['x']) + ',' + str(
            percept['y']) not in self.visitedlocations else None

        np_agentlocs = np.zeros((WORLD_SIZE, WORLD_SIZE), dtype=np.uint8)
        np_breezelocs = np.zeros((WORLD_SIZE, WORLD_SIZE), dtype=np.uint8)
        np_stenchlocs = np.zeros((WORLD_SIZE, WORLD_SIZE), dtype=np.uint8)
        np_visitedlocs = np.zeros((WORLD_SIZE, WORLD_SIZE), dtype=np.uint8)

        np_agentlocs[percept['x'], percept['y']] = 1

        for loc in self.breezelocations:
            np_breezelocs[int(loc.split(',')[0]), int(loc.split(',')[1])] = 1

        if not self.heardscream:
            for loc in self.stenchlocations:
                np_stenchlocs[int(loc.split(',')[0]), int(loc.split(',')[1])] = 1

        for loc in self.visitedlocations:
            np_visitedlocs[int(loc.split(',')[0]), int(loc.split(',')[1])] = 1

        np_breezelocs = np_breezelocs + np.random.rand(np_breezelocs.shape[0], np_breezelocs.shape[1]) / 1000
        np_stenchlocs = np_stenchlocs + np.random.rand(np_stenchlocs.shape[0], np_stenchlocs.shape[1]) / 1000
        np_visitedlocs = np_visitedlocs + np.random.rand(np_visitedlocs.shape[0], np_visitedlocs.shape[1]) / 1000
        np_agentlocs = np_agentlocs + np.random.rand(np_agentlocs.shape[0], np_agentlocs.shape[1]) / 1000

        orient = enc_orientation[percept['direction'].value]
        heardscream = enc_heardscream[self.heardscream]
        glitter = enc_glitter[self.glitter]
        hasgold = enc_hasgold[self.hasgold]
        hasarrow = enc_hasarrow[self.hasarrow]


        state_p1 = np.stack([np_agentlocs, np_stenchlocs, np_breezelocs, np_visitedlocs]).flatten()
        state_p2 = np.append(orient, np.array([heardscream, glitter, hasgold, hasarrow]))
        state = np.append(state_p1, state_p2)

        # return Action.GOFORWARD
        # next_action = random.randrange(0, 6)
        # if next_action == 0: return Action.WALK
        # if next_action == 1: return Action.TURNLEFT
        # if next_action == 2: return Action.TURNRIGHT
        # if next_action == 3: return Action.GRAB
        # if next_action == 4: return Action.SHOOT
        # if next_action == 5: return Action.CLIMB

        return np.array(state)

    def gameOver(self, agent_is_alive):
        """ BeeLineAgent_GameOver: called at the end of each try """
        # print("BeeLineAgent_GameOver: agent alive = {} ".format(agent_is_alive))
        # print(self.escape)