import random
from math import pi, tan, sin, cos
import turtle
from turtle import *

#   ==================================================
#
#                      Need Python2.7
#
#   ==================================================

N = 200    # Number of pedestrians
T = 2500               # Time step. In each steps, they move one step forward or cross

n = 400         # Width of field
m = 400          # Height of field

v = 1.          # Velocity of pedestrians

angle_pi = pi / 3          # Angle of sight
angle_theta = pi / 6       # Angle of movement when pedestrian meet object or other pedestrians

x_info1 = []
y_info1 = []

wn = turtle.Screen()
wn.reset()
#turtle.screensize(n,m)
wn.setworldcoordinates(-5,-5,405,405)
turtlelist = []
turtle.speed(0)
turtle.delay(0)


wn.tracer(3*N, 0)
tantan = tan(angle_pi/2) * 1.0
coscos = cos(angle_theta)* 1.0
sinsin = sin(angle_theta)* 1.0

speedlist = []




def obstacle(i,x,y):
	f = 0
	if i < N / 2 and x < 200:
		r = pow((x-200.),2) + pow((y-300.),2)
		if pow((x-200.),2) + pow((y-300.),2) < 1600:
			f = 1
		
		elif (x-200.)**2 + (y-100.)**2 < 1600:
			f = 1
		
		#if r < 7025 : print x,y,r
	elif i >= N/2 and x > 200:
		r = (x-200.)**2 + (y-300.)**2
		if (x-200.)**2 + (y-300.)**2 < 1600:
			f = 1
		elif (x-200.)**2 + (y-100.)**2 < 1600:
			f = 1		
	return f


 #####obstacle at initializing#####
def obstacle_initial(i,x,y):
	f = 0
	if i < N / 2:
		r = (x-200.)**2 + (y-300.)**2
		if (x-200.)**2 + (y-300.)**2 < 1600:
			f += 1
		elif (x-200.)**2 + (y-100.)**2 < 1600:
			f = 1

	if i >= N/2:
		r = (x-200.)**2 + (y-300.)**2
		if (x-200.)**2 + (y-300.)**2 < 1600:
			f += 1
		elif (x-200.)**2 + (y-100.)**2 < 1600:
			f = 1		
	return f	








#turtle.setup( width = 200, height = 200, startx = None, starty = None)

#   ===================================================
#
#                    SAMPLE GENERATION
#
#   ===================================================

# To get positions of pedestrians and conditions


xx = int(1)
yy = int(1)

for i in range(N) :
	x = 0
	k = 1
	while k == 1:
		x = random.random() * 500 
		y = random.random() * 500 
		k = obstacle_initial(i,x,y) 
		xx += 1
		yy += 1



	turtlelist.append('turtle'+str(i+1))
	x_info1.append(x)
	y_info1.append(y)

	'''
	for j in range(i-1) :
		while x_info1[i] == x_info1[j] and y_info1[i] == y_info1[j]:
			while obstacle_initial(i,x_info1[i],y_info1[i]) == 1:
				x_info1[i] = random.randrange(0,n)
				y_info1[i] = random.randrange(0,m)
	'''

for i in range(N):
	turtlelist[i] = turtle.Turtle()
	if i < N/2:
		turtlelist[i].color("blue")
		turtlelist[i].shape("turtle")
	else:
		turtlelist[i].color("red")
		turtlelist[i].shape("circle")
	turtlelist[i].speed(0)
	
	turtlelist[i].turtlesize(0.4,0.4,0.4)
	turtlelist[i].penup()
	turtlelist[i].setposition(x_info1[i],y_info1[i])





#drawing circle
x = turtle.Turtle()
x.speed("fastest")
x.color("white")
x.goto(200,265)
x.color("black")
x.begin_fill()
x.circle(35)
x.end_fill()
x.color("white")
x.goto(200,65)
x.color("black")
x.begin_fill()
x.circle(35)
x.end_fill()
x.ht()

# Input data in other list for step by step calculation
x_info2 = x_info1
y_info2 = y_info1



#   =====================================================
#
#                     SIMULATION PROCESS
#
#   =====================================================

# Do Simulation process
# Compare target pedestrian p with others who comes opposite direction
# This version not include obstacle
for l in range(T) :

	#synchronous update#
	x_info1 = x_info2
	y_info1 = y_info2
	right_std = 0
	right_irr = 0 
	left_std = 0
	left_irr = 0
	for p in range(N) : #for every agents
			# irr is meet event value. If irr > 0, pedestrian do avoiding movement.

		std = 0 
		irr = 0
		stopup = 0






		#print f



		########### obstacle move ############
		if obstacle(p,x_info1[p],y_info1[p]) != 0:
			
			if p < N/2:
				x_info2[p] = x_info1[p] - 4*v*coscos
				y_info2[p] = y_info1[p] + 4*v*sinsin
				turtlelist[p].setpos(x_info2[p],y_info2[p])
				right_irr += 1
				continue
			if p >= N/2:
				x_info2[p] = x_info1[p] + 4*v*coscos
				y_info2[p] = y_info1[p] - 4*v*sinsin
				turtlelist[p].setpos(x_info2[p],y_info2[p])
				left_irr += 1
				continue				



		if p < N/2:

			for q in range(0, N):


				if x_info1[q] <= x_info1[p] - 2:
					continue

				dx = x_info1[q] - x_info1[p]
				dy = y_info1[q] - y_info1[p]
				dist = (dx**2 + dy**2)**0.5

				if dist > 12:
					continue
				

				if dist < 12. and dx != 0 and abs(dy/dx) < tantan and q > N/2: 
					irr += 1

				if dist < 2. and dx != 0 and abs(dy/dx) < tantan and q <= N/2: 
					irr += 1	

				if irr > 0 and y_info1[q] - 1 > y_info1[p] and dx != 0:
					if dist < 1. and abs(dy/dx) > tantan: 
						stopup += 1


					
			

			if irr > 0 and stopup < 1:
				x_info2[p] = x_info1[p] - 4*v*coscos
				y_info2[p] = y_info1[p] + 4*v*sinsin
				right_irr += 1

			
			elif stopup < 1:
				x_info2[p] = x_info1[p] + v
				right_std += 1

		elif p >= N/2 :

			for q in range(0, N) :
				

				if x_info1[p] + 2 <= x_info1[q] : 
					continue

				dx = x_info1[p] - x_info1[q]
				dy = y_info1[p] - y_info1[q]
				dist = (dx**2 + dy**2)**0.5
				
				if dist > 12:
					continue
				

		
				if dist < 12. and dx != 0 and abs(dy/dx) < tantan and q < N/2: 
					irr += 1
				if dist < 2. and dx != 0 and abs(dy/dx) < tantan and q >= N/2: 
					irr += 1					
				if irr > 0 and y_info1[q]  < y_info1[p] - 1 and dx != 0 :
					if dist < 1. and abs(dy/dx) > tantan:	
						stopup += 1



			if irr > 0 and stopup < 1:
				x_info2[p] = x_info1[p] + 4*v*coscos
				y_info2[p] = y_info1[p] - 4*v*sinsin
				left_irr += 1


			elif stopup < 1:
				x_info2[p] = x_info1[p] - v
				left_std += 1


		

		#######------Boundary Condition------########
		if x_info2[p] > n : x_info2[p] = x_info2[p] - n
		if x_info2[p] < 0 : x_info2[p] = x_info2[p] + n
		if y_info2[p] > m : y_info2[p] = y_info2[p] - m
		if y_info2[p] < 0 : y_info2[p] = y_info2[p] + m


		turtlelist[p].setpos(x_info2[p],y_info2[p])

	a = right_std + left_std - 4*coscos*(right_irr + left_irr)

	speedlist.append(a/N)




	#####------plot------#####
	# See the plot in each 100 times

turtle.mainloop()
