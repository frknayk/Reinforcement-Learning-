import gym
import numpy as np
import random
from math import pow
from math import sqrt
from math import exp
import pandas
import matplotlib.pyplot as plt

class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        # Defining Q table as dictionary,
        # provides us to empty state-space
        # and any unvisited-states
        self.q = {}

        # exploration constant
        self.epsilon = epsilon  

        # Learning Rate : Alpha
        # How is new value estimate weighted against the old (0-1).
        #  1 means all new and is ok for no noise situations.
        self.alpha = alpha     

        # Discount Factor : Gamma 
        # When assessing the value of a state & action,
        # how important is the value of the future states
        self.gamma = gamma      

        self.actions = actions

    def getQ(self, state, action):
        return self.q.get((state, action), 0.0)

    def learnQ(self, state, action, reward, value):
        #    Q(s, a) += alpha * (reward(s,a) + max(Q(s') - Q(s,a))
        oldv = self.q.get((state, action), None)
        if oldv is None:
            self.q[(state, action)] = reward
        else:
            self.q[(state, action)] = oldv + self.alpha * (value - oldv)

    def chooseAction(self, state):
        q = [self.getQ(state, a) for a in self.actions]
        maxQ = max(q)

        if random.random() < self.epsilon:
            minQ = min(q); mag = max(abs(minQ), abs(maxQ))
            # add random values to all the actions, recalculate maxQ
            q = [q[i] + random.random() * mag - .5 * mag for i in range(len(self.actions))]
            maxQ = max(q)

        count = q.count(maxQ)
        # In case there're several state-action max values
        # we select a random one among them
        if count > 1:
            best = [i for i in range(len(self.actions)) if q[i] == maxQ]
            i = random.choice(best)
        else:
            i = q.index(maxQ)

        action = self.actions[i]

        return action

    def learn(self, state1, action1, reward, state2):
        maxqnew = max([self.getQ(state2, a) for a in self.actions])
        self.learnQ(state1, action1, reward, reward + self.gamma*maxqnew)

def build_state(features):
    return int("".join(map(lambda feature: str(int(feature)), features)))

def to_bin(value, bins):
    return np.digitize(x=[value], bins=bins)[0]


################################################## Training Phase ##################################################

env = gym.make('CartPole-v0')

MAX_EPISODE = 3100
MAX_STEP = 200
n_bins_cart_pos = 14
n_bin_cart_vel = 12
n_bins_angle = 10
n_bins_angle_vel = n_bins_angle
cart_position_save = {}
pole_angle_save = {}
control_signal_save = {}
number_of_features = env.observation_space.shape[0]

# Number of states is huge so in order to simplify the situation
# we discretize the space to: 10 ** number_of_features
cart_position_bins = pandas.cut([-2.4, 2.4], bins=n_bins_cart_pos, retbins=True)[1][1:-1]
cart_velocity_bins = pandas.cut([-1, 1], bins=n_bin_cart_vel, retbins=True)[1][1:-1]
pole_angle_bins = pandas.cut([-2, 2], bins=n_bins_angle, retbins=True)[1][1:-1]
angle_rate_bins = pandas.cut([-3.5, 3.5], bins=n_bins_angle, retbins=True)[1][1:-1]

# The Q-learn algorithm
qlearn = QLearn(actions=range(env.action_space.n),
                    alpha=0.5, gamma=0.90, epsilon=0.2)

def reward_func_single(state) :
    rew_coeff_x  = 1
    rew_coeff_xdot  = 3
    rew_coeff_theta  = 1
    rew_coeff_thetadot  = 1

    # print(state[0],state[1],state[2],state[3])

    error_x = rew_coeff_x*np.abs(state[0])
    error_xdot = rew_coeff_xdot*np.abs(pow(state[1],2))
    error_theta = rew_coeff_theta*np.abs(state[2])
    error_thetadot = rew_coeff_thetadot*np.abs( pow(state[3],2))
    
    error=[]
    error.append(error_x)
    error.append(error_xdot)
    error.append(error_theta)
    error.append(error_thetadot)
    max_e = max(error)
    min_e = min(error)
    dif = max_e - min_e
    for x in range(len(error)):
        error[x] = (error[x] - min_e)/(dif)
    return sum(error)
    # error_all = error_x+error_xdot+error_theta+error_thetadot
    # return error_all

def avg(lst): 
    return sum(lst) / len(lst) 

for i_episode in range(MAX_EPISODE):

    observation = env.reset()
    # x,x_dot,theta,theta_dot 
    cart_position,cart_velocity, pole_angle, angle_rate_of_change = observation

    state = build_state([to_bin(cart_position, cart_position_bins),
    to_bin(cart_velocity, cart_velocity_bins),
    to_bin(pole_angle, pole_angle_bins),
    to_bin(angle_rate_of_change, angle_rate_bins)])
    
    local_cart_position = []
    local_pole_angle = []
    local_control_signal = []

    for t in range(MAX_STEP):
        if( i_episode>MAX_EPISODE*0.95) : env.render()
        
        #  Pick an action based on the current state
        action = qlearn.chooseAction(state)
        cs = 0
        if action==0 :
            cs = -10
        else :
            cs = 10


        # Execute the action and get feedback
        observation, reward, done, info = env.step(action)
        # Digitize the observation to get a state
        cart_position,cart_velocity, pole_angle, angle_rate_of_change = observation
        
        local_cart_position.append(cart_position)
        local_pole_angle.append(pole_angle)
        local_control_signal.append(cs)
        # print(reward_func_single( observation) )
        rewcard = reward_func_single(observation)
        nextState = build_state([to_bin(cart_position, cart_position_bins),
        to_bin(cart_velocity, cart_velocity_bins),
        to_bin(pole_angle, pole_angle_bins),
        to_bin(angle_rate_of_change, angle_rate_bins)])
        if not(done):
            qlearn.learn(state, action, reward, nextState)
            qlearn.epsilon = qlearn.epsilon * 0.999 # added epsilon decay
            state = nextState
        else:
            reward = -200
            qlearn.learn(state, action, reward, nextState)
            qlearn.epsilon = qlearn.epsilon * 0.999 # added epsilon decay

            break
    
    print("Episode : #{0} was #{1} steps.".format(i_episode,t))
    # Calculate mean cart position and pole angle for the episode
    cart_position_save[i_episode] = avg(local_cart_position)
    pole_angle_save[i_episode] = avg(local_pole_angle)
    control_signal_save[i_episode] = avg(local_control_signal)

print("Length of State-Space : ",len(qlearn.q))
# time = np.linspace(0,MAX_EPISODE,MAX_EPISODE)
position_vals = list(cart_position_save.values())
angle_vals = list(pole_angle_save.values())
control_vals = list(control_signal_save.values())

# Saving Position, Angular Position and Control Signal Values
# And plotting later in 'q_learn_inference_bin.py'
np.save('CartPosByTime.npy',position_vals)
np.save('AngularPosByTime.npy',angle_vals)
np.save('ControlSignalByTime.npy',control_vals)


np.save("policy_bin",qlearn.q)