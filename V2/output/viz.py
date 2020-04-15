import matplotlib.pyplot as plt
from numpy import linspace
from numpy import flipud
from numpy import floor

file = open("trainVanillaWithCommentsAndModifications.log",'r')
errors = []
bno = []
for line in file.readlines():
	errors.append(float(line.split(" ")[1]))
	bno.append(line.split()[0])

plt.plot(bno, errors)
plt.title("Error Curve")
plt.xlabel("Gradient Steps")
plt.ylabel("Error")
#plt.show()
plt.savefig("ErrorCurve.png")
