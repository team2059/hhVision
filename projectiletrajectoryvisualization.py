from matplotlib import pyplot as plt
import math
import numpy as np
import cv2 as cv

def nothing(): pass

cv.namedWindow('params')
cv.createTrackbar("theta", 'params', 45, 90, nothing)
cv.createTrackbar("v_0", 'params', 200, 200, nothing)
cv.createTrackbar("goal_x", 'params', 100, 100, nothing)

maxtime = 2.0 #seconds

g = -9.81 #meters/s^2
dt = .001 #seconds
distance_uncertainty = .333333 #meters
goal_y_min = 3 #meters
goal_y_max = 3.5 #meters
goal_y_avg = .5*goal_y_min+.5*goal_y_max #meters

while(1):
	if cv.waitKey(1) & 0xFF == ord('q'):
		break

	goal_x = cv.getTrackbarPos("goal_x", 'params')/15 #meters
	
	#theta = cv.getTrackbarPos("theta", 'params') #degrees
	theta = math.degrees(math.atan(2*goal_y_avg/goal_x)) #degrees

	#v_0 =  cv.getTrackbarPos("v_0", 'params')/10.0 #meters/s
	v_0 = math.sqrt((2*-g*goal_x/math.sin(math.radians(2*theta)))) #meters/s
	
	print("goal distance:", goal_x, theta, v_0)

	time = []
	x_pos = []
	y_pos = []
	x_vel = []
	y_vel = []
	
	xvel = v_0*math.cos(math.radians(theta)) #meters/s
	xpos = 0 #meters
	yvel = v_0*math.sin(math.radians(theta)) #meters/s
	ypos = 0 #meters
	t = 0 #seconds
	
	goalPassed = False
	while(t <= maxtime and ypos>=0 and not goalPassed):
		xpos+=xvel*dt
		yvel+=g*dt
		ypos+=yvel*dt
		
		if(abs(xpos-(goal_x+distance_uncertainty))<.01):
			goalPassed = True
		
		x_pos.append(xpos)
		y_pos.append(ypos)
		time.append(t)
		t+=dt
	
	plt.plot(x_pos, y_pos, "b-", label = "trajectorty")
	plt.plot([goal_x-distance_uncertainty, goal_x-distance_uncertainty], [goal_y_min,goal_y_max], "r--", label = "goal closest")
	plt.plot([goal_x, goal_x], [goal_y_min,goal_y_max], "g-", label = "true goal distance")
	plt.plot([goal_x+distance_uncertainty, goal_x+distance_uncertainty], [goal_y_min,goal_y_max], "c--", label = "goal farthest")
	
	plt.xlim(0, 10)
	plt.ylim(0, max(max(y_pos), goal_y_max))
	plt.draw()
	plt.grid(True)
	plt.legend()
	plt.pause(.001)
	plt.cla()
