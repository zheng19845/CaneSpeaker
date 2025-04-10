''' Room-to-Room navigation environment '''
import sys
sys.path.append('./build')

import json
import MatterSim
import numpy as np
import math
import cv2
from PIL import Image
import networkx as nx
import random

_action_space = {
    'forward' : "forward",
    'left' : "left",
    'right' : "right",
    'up' : "up",
    'down' : "down",
    'stop' : "stop" 
}



def load_nav_graphs(scans, root_dir = r"./"):
    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3]-pose2['pose'][3])**2\
          + (pose1['pose'][7]-pose2['pose'][7])**2\
          + (pose1['pose'][11]-pose2['pose'][11])**2)**0.5
        
    graphs = {}
    for scan in scans:
        with open(root_dir + 'connectivity/%s_connectivity.json' % scan) as f:
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            for i,item in enumerate(data):
                if item['included']:
                    for j,conn in enumerate(item['unobstructed']):
                        if conn and data[j]['included']:
                            positions[item['image_id']] = np.array([item['pose'][3],
                                    item['pose'][7], item['pose'][11]]);
                            assert data[j]['unobstructed'][i], 'Graph should be undirected'
                            G.add_edge(item['image_id'],data[j]['image_id'],weight=distance(item,data[j]))
            nx.set_node_attributes(G, values=positions, name='position')
            graphs[scan] = G
    return graphs



class Env():
    ''' A simple wrapper for a batch of MatterSim environments,
        using discretized viewpoints and pretrained features '''

    def __init__(self):
        WIDTH = 640
        HEIGHT = 480
        VFOV = math.radians(105)

        self.sim = MatterSim.Simulator()
        # self.sim.setRenderingEnabled(False)
        self.sim.setCameraResolution(WIDTH, HEIGHT)
        self.sim.setCameraVFOV(VFOV)
        self.sim.setDiscretizedViewingAngles(True)
        self.sim.setDepthEnabled(False) # Turn on depth only after running ./scripts/depth_to_skybox.py (see README.md)
        self.sim.initialize()
        
    def _make_id(self, scanId, viewpointId):
        return scanId + '_' + viewpointId

    def newEpisode(self, scanId, viewpointId, heading):
        self.sim.newEpisode([scanId], [viewpointId], [heading], [0])

    def getState(self):
        return self.sim.getState()[0]

    def makeAction(self, action):
        ix = [int(action[0])]
        heading = [float(action[1])]
        elevation = [float(action[2])]
        self.sim.makeAction(ix, heading, elevation)

    def makeSimpleActions(self, simple_action):
        ''' Take an action using a simple interface: 0-forward, 1-turn left, 2-turn right, 3-look up, 4-look down.
            All viewpoint changes are 30 degrees. Forward, look up and look down may not succeed - check state.
            WARNING - Very likely this simple interface restricts some edges in the graph. Parts of the
            environment may not longer be navigable. '''

        if simple_action == _action_space['forward']:
            action=[1, 0, 0]
        elif simple_action == _action_space['left']:
            action=[0,-math.pi/6.0, 0]
        elif simple_action == _action_space['right']:
            action=[0, math.pi/6.0, 0]
        elif simple_action == _action_space['up']:
            action=[0, 0, math.pi/6.0]
        elif simple_action == _action_space['down']:
            action=[0, 0,-math.pi/6.0]
        else:
            print("Invalid simple action")
            return False
        self.makeAction(action)
        return True
    
    
class R2R_Planner():
    def __init__(self, scans, root_dir):
        self.scans = scans
        self.graphs = load_nav_graphs(scans, root_dir)
        self.paths = {}
        for scan,G in self.graphs.items(): # compute all shortest paths
            self.paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
        self.env = Env()
        pass
    
    def _shortest_path_action(self, state, goalViewpointId):
        ''' Determine next action on the shortest path to goal, for supervised training. '''
        if state.location.viewpointId == goalViewpointId:
            return "stop", 1 # do nothing
        path = self.paths[state.scanId][state.location.viewpointId][goalViewpointId]
        nextViewpointId = path[1]
        for i,loc in enumerate(state.navigableLocations):
            if loc.viewpointId == nextViewpointId:
                # print("Found in navigable locations")
                # Look directly at the viewpoint before moving
                if loc.rel_heading > math.pi/12.0+1e-2:
                    # print("Checkpoint1")
                    return "right", 1 # Turn right
                elif loc.rel_heading < -math.pi/12.0-1e-2:
                    # print("Checkpoint2")
                    return "left", 1 # Turn left
                elif loc.rel_elevation > math.pi/12.0+1e-2 and state.viewIndex//12 < 2:
                    # print("Checkpoint3")
                    return "up", 1 # Look up
                elif loc.rel_elevation < -math.pi/12.0-1e-2 and state.viewIndex//12 > 0:
                    # print("Checkpoint4")
                    return "down", 1 # Look down
                else:
                    return i, 1 # Move
        # Can't see it - first neutralize camera elevation
        if state.viewIndex//12 == 0:
            return "up", 0 # Look up
        elif state.viewIndex//12 == 2:
            return "down", 0 # Look down

        # Otherwise decide which way to turn
        pos = np.array([state.location.x, state.location.y, state.location.z])
        target_rel = self.graphs[state.scanId].nodes[nextViewpointId]['position'] - pos
        target_heading = math.pi/2.0 - math.atan2(target_rel[1], target_rel[0]) # convert to rel to y axis
        if target_heading < 0:
            target_heading += 2.0*math.pi
        # print(state.heading, target_heading)
        if state.heading > target_heading and state.heading - target_heading < math.pi:
            # print("Checkpoint5")
            return "left", 0 # Turn left
        if target_heading > state.heading and target_heading - state.heading > math.pi:
            # print("Checkpoint6")
            return "left", 0 # Turn left
        # print(state.heading/math.pi*180, target_heading/math.pi*180)
        # print(target_rel)
        # print(state.location.viewpointId)
        # print(nextViewpointId)
        # print([loc.viewpointId for _,loc in enumerate(state.navigableLocations)])
        return "right", 0 # Turn right
   
     
    def generate_actions(self, item, visualize = False):
            scan = item["scan"]
            path = item["path"]
            self.env.newEpisode(scan, path[0],item["heading"])
            seq = []
            mask = []
            i = 0

            count = 0
            while True:
                state = self.env.getState()
                count = count + 1
                
                
                if i == len(path)-1:
                    rgb = np.array(state.rgb, copy=False)
                    if visualize:
                        cv2.imshow("RGB", rgb)
                        cv2.waitKey(0)
                    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                    seq.append(Image.fromarray(rgb))
                    mask.append(0)
                    break
                
                action, _ = self._shortest_path_action(state, path[i+1])
                
                if action == 'stop':
                    i+=1
                    continue
                
                rgb = np.array(state.rgb, copy=False)
                if visualize:
                    cv2.imshow("RGB", rgb)
                    cv2.waitKey(0)
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                seq.append(Image.fromarray(rgb))
                mask.append(0)
                
                if isinstance(action, int):
                    seq.append(_action_space["forward"])
                    mask.append(1)
                    self.env.makeAction([action,0,0])
                else:
                    seq.append(_action_space[action])
                    mask.append(1)
                    self.env.makeSimpleActions(_action_space[action])             
                    
            seq.append(_action_space["stop"])
            mask.append(1)
            
            # shorten            
            action_list = []
            image_list = []
            for i in range(len(mask)):
                if mask[i]==0:
                    image_list.append(seq[i])
                elif mask[i]==1:
                    action_list.append(seq[i])
            
            new_action = []
            new_image = []
            
            i=0
            while i<len(action_list):
                j=i
                while j<len(action_list) and action_list[j]==action_list[i]:
                    j+=1
                    
                length = j-i
                
                if action_list[i]==_action_space["forward"]:

                    # cur_action = action_list[i:j]
                    # tmp = [cur_action[k:k+2] for k in range(0,j-i,2)]
                    # new_action+=[", ".join(tmp0) for tmp0 in tmp]
                    
                    # new_action+=action_list[i:j:2]
                    # new_image+=image_list[i:j:2]
                    new_action+=action_list[i:j]
                    new_image+=image_list[i:j]
                else:
                    if length in [1,2,3]:
                        # new_action.append(", ".join(action_list[i:j]))
                        new_action.append(action_list[i])
                        new_image.append(image_list[i])
                    elif length in [4,5,6]:
                        d = (length+1)//2
                        # new_action+=[", ".join(action_list[i:i+d]), ", ".join(action_list[i+d:j])]
                        new_action+=action_list[i:j:d]
                        new_image+=image_list[i:j:d]
                        
                i=j
                
            seq = []
            mask = []

                
            for i in range(len(new_action)):
                seq += [new_image[i], new_action[i]]
                mask += [0,1]
            # print(seq)                
            if item['type'] == 'SOON' or item['type'] == 'REVERIE':
                if random.randint(0,1):
                    action = 'left'
                else:
                    action = 'right'
                num = random.choice([4, 6])
                for view_num in range(num - 1):
                    # print(view_num)
                    for turn_num in range(int(12 / num)):
                        # print(turn_num)
                        self.env.makeSimpleActions(_action_space[action])
                    # print()
                    state = self.env.getState()
                    # print(state.viewIndex)
                    rgb = np.array(state.rgb, copy=False)
                    if visualize:
                        cv2.imshow("RGB", rgb)
                        cv2.waitKey(0)
                    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                    seq.append(Image.fromarray(rgb))
                    mask.append(0)
                    
                    # seq.append(", ".join([_action_space[action]]*int(12 / num)))
                    seq.append(_action_space[action])
                    mask.append(1)
            # print(seq)
            return seq, mask
            
