import db_data
import pitch_data
import numpy as np
from sklearn.linear_model import LinearRegression
import json
import matplotlib.pyplot as plt

db = json.loads(db_data.calculate_db("preamble.wav"))
#throw away first and last second
del db[0][:10]
del db[0][-10:]

#scatter plot of db values
plt.plot(list(range(0, len(db[0]))), db[0], 'o')
#linear regresssion on db values
m, b = np.polyfit(list(range(0, len(db[0]))), db[0], 1)
#plot linreg
plt.plot(list(range(0, len(db[0]))), list(m * range(0, len(db[0])) + b))
print("db linear regression coefficient: " + str(m))
plt.show()


# pitch = json.loads(pitch_data.pitch_magnitudes("louder.wav"))
# plt.plot(list(range(0, len(pitch[0]))), pitch[0], 'o')
# print(pitch)
# plt.show()
