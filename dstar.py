import math
import numpy as np
from sys import maxsize


class State:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.state = "."
        self.t = "new"  # tag for state
        self.h = 0
        self.k = 0
        self.action = -1

    def cost(self, state):
        if self.state == "#" or state.state == "#":
            return maxsize

        return math.sqrt(math.pow((self.x - state.x), 2) +
                         math.pow((self.y - state.y), 2))

    def set_state(self, state):
        """
        .: new
        #: obstacle
        e: oparent of current state
        *: closed state
        s: current state
        """
        if state not in ["s", ".", "#", "e", "*"]:
            return
        self.state = state


class Map:

    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.map = self.init_map()

    def init_map(self):
        map_list = []
        for i in range(self.row):
            tmp = []
            for j in range(self.col):
                tmp.append(State(i, j))
            map_list.append(tmp)
        return map_list

    def get_neighbors(self, state):
        state_list = []
        for i in [1, 0, -1]:
            for j in [1, 0, -1]:
                if i == 0 and j == 0:
                    continue
                if state.x + i < 0 or state.x + i >= self.row:
                    continue
                if state.y + j < 0 or state.y + j >= self.col:
                    continue
                # if self.map[state.x + i][state.y + j].state == "#":
                #     continue
                state_list.append(self.map[state.x + i][state.y + j])

        return state_list

    def set_obstacle(self, point_list):
        for y, x in point_list:
            # x = 20 - x
            if x < 0 or x >= self.row or y < 0 or y >= self.col:
                continue

            self.map[x][y].set_state("#")

    
    def get_map(self, prize_locations, agents_locations=None):
        map = np.array(['' for _ in range(self.row)]*self.col).reshape(self.row,self.col)
        for i in range(self.row):
            for j in range(self.col):
                if self.map[i][j].state == "#":
                    map[i][j] = '|'

        for x, y in prize_locations:
            map[int(x)][int(y)] = 'x'

        # for x, y in agents_locations:
        #     map[int(x)][int(y)] = 'o'

        print ("self.map: \n", map)


class Dstar:
    def __init__(self, maps):
        self.map = maps
        self.open_list = set()
        self.action_dict = {(1,0):0, (-1,0):1, (0,1):2, (0,-1):3, (1,1):4, (1,-1):5, (-1,1):6, (-1,-1):7}

    def process_state(self):
        x = self.min_state()

        if x is None:
            return -1

        k_old = self.get_kmin()
        self.remove(x)
        state_list = self.map.get_neighbors(x)
        if k_old < x.h:
            for y in state_list:
                if y.h <= k_old and x.h > y.h + x.cost(y):
                    x.parent = y
                    x.h = y.h + x.cost(y)
        elif k_old == x.h:
            for y in state_list:
                if y.t == "new" or y.parent == x and y.h != x.h + x.cost(y) \
                        or y.parent != x and y.h > x.h + x.cost(y):
                    y.parent = x
                    self.insert(y, x.h + x.cost(y))
        else:
            for y in state_list:
                if y.t == "new" or y.parent == x and y.h != x.h + x.cost(y):
                    y.parent = x
                    self.insert(y, x.h + x.cost(y))
                else:
                    if y.parent != x and y.h > x.h + x.cost(y):
                        self.insert(y, x.h)
                    else:
                        if y.parent != x and x.h > y.h + x.cost(y) \
                                and y.t == "close" and y.h > k_old:
                            self.insert(y, y.h)
        return self.get_kmin()

    def min_state(self):
        if not self.open_list:
            return None
        min_state = min(self.open_list, key=lambda x: x.k)
        return min_state

    def get_kmin(self):
        if not self.open_list:
            return -1
        k_min = min([x.k for x in self.open_list])
        return k_min

    def insert(self, state, h_new):
        if state.t == "new":
            state.k = h_new
        elif state.t == "open":
            state.k = min(state.k, h_new)
        elif state.t == "close":
            state.k = min(state.h, h_new)
        state.h = h_new
        state.t = "open"
        self.open_list.add(state)

    def remove(self, state):
        if state.t == "open":
            state.t = "close"
        self.open_list.remove(state)

    def modify_cost(self, x):
        if x.t == "close":
            self.insert(x, x.parent.h + x.cost(x.parent))

    def run(self, start, end):
        pos_list = []
        action_list = []
        rx = []
        ry = []
        iteration = 0
        feasible = True

        self.open_list.add(end)
        iteration2 = 0
        while True and iteration2 <200:
            iteration2 +=1
            self.process_state()
            #print("whileda donuyo")
            if start.t == "close":
                #print("ife girdi")
                break
        if iteration2>=200:    
            return False, pos_list, action_list
        
        start.set_state("s")
        s = start
        s = s.parent
        s.set_state("e")
        tmp = start

        while tmp != end and iteration < 100:
            iteration += 1
            tmp.set_state("*")
            rx.append(tmp.x)
            ry.append(tmp.y)
            pos_list.append([tmp.x,tmp.y])
            # if show_animation:
            #     plt.plot(rx, ry, "-r")
            #     plt.pause(0.01)
            if tmp.parent.state == "#":
                feasible = self.modify(tmp)
                continue
            tmp = tmp.parent
            
        if tmp == end:
            # print ("iter: ", iteration)
            tmp.set_state("e")            
            pos_list.append([end.x,end.y])
            pos_list = np.array(pos_list)
            for ind in range(len(pos_list) - 1):
                diff = pos_list[ind + 1] - pos_list[ind]
                action_list.append(self.action_dict[tuple(diff)])
        else:
            # print ("Infeasible path planning!")
            feasible = False
        
        return feasible, pos_list, action_list

    

    def modify(self, state):
        self.modify_cost(state)
        iteration = 0
        while True and iteration <200:
            iteration += 1
            k_min = self.process_state()
            if k_min >= state.h:
                break
        return False if iteration>=200 else True


    def test_models(self): # trivial function
        for name in glob.glob(os.path.join(args.load_model, '*.pth')):
            mean_reward = 0
            if name.find('policy') > 0:
                model_no = (int(re.findall(r'\d+', name)[0]))
                dqn.load_models(args.load_model, model_no)

                for i_iter in range(args.test_iteration):
                    seed_number = i_iter % 20
                    agent_obs = env.reset(seed_number)
                    episode_reward = 0

                    action = dqn.choose_action(agent_obs) # output is between 0 and 7
                    n_agents = action + 1 # number of allowable agents is 1 to 8
                    episode_reward, done, agent_next_obs = env.step(n_agents)

                    #print('Episode: ', i_iter + 1, '| Episode Reward: ', round(episode_reward, 2))

                    mean_reward += episode_reward

                mean_reward = mean_reward / args.test_iteration
                #print('Model: {0} / Mean Reward: {1:.3} \n'.format(model_no, mean_reward))
                model_reward_list[model_no] = mean_reward

                with open('model_reward_list.pkl', 'wb') as f:  
                    pickle.dump(model_reward_list, f)
