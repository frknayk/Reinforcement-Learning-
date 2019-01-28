import numpy as np
import matplotlib.pyplot as plt

cartPos = np.load('CartPosByTime.npy')
PoleAngle = np.load('AngularPosByTime.npy')
control_vals = np.load('ControlSignalByTime.npy')

# Plotting
plt.figure(1)
plt.plot(cartPos, label='Cart Position')
plt.plot(PoleAngle, label='Anglular Position')
plt.legend()


plt.figure(2)
plt.subplot(211)
plt.plot(control_vals, label='Control Signal')
plt.legend()
plt.show()

print(control_vals)