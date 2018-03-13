import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op

def func1(x):
    return ((x-2)**2 +2)
def derivf1(x):
    return (2*(x-2))
def funcRose(x, y):
    return (1-x)**2 + 100*(y-(x)**2)**2
def derivRose(x, y):
    return [2*(1-x) + 100*2*-2*x*(y-x**2), 100*2*(y-x**2)]
#
xx, yy= np.linspace(-3, 3, 100), np.linspace(-3, 3, 100)
X, Y= np.meshgrid(xx, yy)
Z= funcRose(X, Y)

plt.contour(X, Y, Z, colors= 'gray', levels= [0.7, 3, 5, 15, 50, 150, 500, 1500, 5000])
alpha= 4e-04
startx, starty= -2.5, -2.5
plt.plot(startx, starty, 'go', markersize= 7)
for i in range(50):
    startx, starty= startx - alpha*derivRose(startx, starty)[0], starty - alpha*derivRose(startx, starty)[1]
    plt.plot(startx, starty, 'go', markersize= 7)
    print(startx, starty)


# print(X, Y)
# plt.plot(xx, func1(xx), 'k-')
# plt.xlim(-5, 5)
#
# startx= -3
#
# for i in range(50):
#     plt.plot(startx, func1(startx), 'go', markersize= 7)
#     startx= startx- 0.1*derivf1(startx)
#     print("x= {}, y={}".format(startx, func1(startx)))
#     i+=1
#
plt.show()

# result= op.minimize(func1, 1)
# print(result)
