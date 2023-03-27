import numpy as np
import matplotlib.pyplot as plt
import random as rd

n_samples = 10

w2 = 1/2
w1 = 3
w0 = 10

x_data = np.sort(np.random.uniform(-15, 10, n_samples))
y_data = (w2 * x_data ** 2) + (w1 * x_data) + (w0) + np.random.normal(0, 4, n_samples)

for i in range(1, 8 + 1, 1):

  p = i

  H = np.zeros((n_samples, i + 1))

  for j in range(p, 0 - 1, -1):
    H[:, p - j] = x_data ** j
  
  H_pinv = np.linalg.pinv(H)

  w = H_pinv @ np.asmatrix(y_data).T

  x_eq = np.arange(-15, 10, 0.1)

  y_eq_in = (w2 * x_eq ** 2) + (w1 * x_eq) + (w0)

  y_eq_out = np.zeros(len(x_eq))

  for j in range(p, 0 - 1, -1):
    y_eq_out = y_eq_out + (w[p - j] * x_eq ** j)

  y_eq_out = y_eq_out.T
  # y_eq_out = ((w[0] * x_eq ** 2) + (w[1] * x_eq) + w[2]).T

  fig, axs = plt.subplots(2, 2)
  fig.set_figwidth(10)
  fig.set_figheight(8)
  fig.suptitle("Approximation using "+ str(i) + " degree polynomial")

  axs[0, 0].plot(x_eq, y_eq_in, '-b', linewidth=0.5)
  axs[0, 0].plot(x_data, y_data, 'or', mfc='none')
  axs[0, 0].set_title("Original function + simulated data")

  axs[0, 1].plot(x_eq, y_eq_out, '-g', linewidth=0.5)
  axs[0, 1].plot(x_data, H @ w, 'og', mfc='none')
  axs[0, 1].set_title("Adjusted function + adjusted data")

  axs[1, 0].plot(x_eq, y_eq_in, '-b', linewidth=0.5)
  axs[1, 0].plot(x_eq, y_eq_out, '-g', linewidth=0.5)
  axs[1, 0].set_title("Original function + adjusted function")

  axs[1, 1].plot(x_eq, y_eq_in, '-b', linewidth=0.5)
  axs[1, 1].plot(x_eq, y_eq_out, '-g', linewidth=0.5)
  axs[1, 1].plot(x_data, y_data, 'or', mfc='none')
  axs[1, 1].plot(x_data, H @ w, 'og', mfc='none')
  axs[1, 1].set_title("All together")

  # plt.show()
  plt.savefig("fig" + str(i))

# print(values)
