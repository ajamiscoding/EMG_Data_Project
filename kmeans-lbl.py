
import numpy as np
import matplotlib.pyplot as plt
x = [0, 47, 79, 110, 141, 172, 204, 250, 266, 313, 360, 391, 422, 454, 500]
Y = [0, -1, -2, -1, -2, -2, -1, -2, 0, 0, -3, 0, 0, 0, 0]
from scipy.interpolate import interp1d
f = interp1d(x, Y, kind='cubic', fill_value='extrapolate')
# Generate a smooth curve using the interpolated function
x_new = np.linspace(0, 500, 500)
Y_new = f(x_new)

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(Y_new.reshape(-1, 1))
clusters=kmeans.labels_
plt.plot(x, Y, 'o', label='data points')
plt.scatter(x_new, Y_new,c=clusters)
plt.show()