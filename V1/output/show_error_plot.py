import matplotlib.pyplot as plt
from numpy import linspace
from numpy import flipud
from numpy import floor
from numpy import random

file = open("trainLog.log","r")
errors = []
validation = []
bno = []
count = 0
cross_over = [[],[]]
for line in file.readlines():
	errors.append(float(line.split(" ")[1]) )
	validation.append(float(line.split(" ")[2]))
	bno.append(line.split()[0])

plt.plot(bno, errors,label = 'Train Loss')
plt.plot(bno, validation, label = 'Validation Loss' )
plt.title("Error Curve")
plt.xlabel("Gradient Steps")
plt.ylabel("Error")
plt.legend()
plt.xticks(range(0, len(bno),100))
plt.scatter(cross_over[1], cross_over[0])
plt.show()
