import numpy as np
import matplotlib.pyplot as plt

def moving_average(y, w):
  y_padded = np.pad(y, (w//2, w-1-w//2), mode='edge')
  return np.convolve(y_padded, np.ones((w,))/w, mode='valid') 

def plot_rewards(file, title):
  data = np.load(file)
  mov_avg = moving_average(data, 100)
  x = np.arange(0,len(data))
  plt.plot(x, data, label="return")
  plt.plot(x, mov_avg, label="moving average")

  plt.xlabel("episodes")
  plt.ylabel("R")
  plt.legend()
  plt.title(title)

plt.subplots(figsize=(20,20))

plt.subplot(2,2,1)
plot_rewards('outputs/1a/rewards.npy', 'task 1a')

plt.subplot(2,2,2)
plot_rewards('outputs/1b/rewards.npy', 'task 1b')

plt.subplot(2,2,3)
plot_rewards('outputs/1c/rewards.npy', 'task 1c')

plt.subplot(2,2,4)
plot_rewards('outputs/2a/rewards.npy', 'task 2a')

plt.show()
