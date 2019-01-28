# Reinforcement-Learning-
Reinforcement learning implementations mostly on Open AI's Cart Pole environment

I aimed to stabilize the pendulum on the (x,x',theta,theta') = (0,0,0,0) state. 

Achieving satisfying results highly depends on discretization of the state space. I used 14 bins for position state and 12 for velocity state. Velocity changes quite faster than other states, so you may really consider discretize this state more than angular velocity.

In the inference file, you can see the performance of the algorithm by change of control signal, cart position and pole angle. 
