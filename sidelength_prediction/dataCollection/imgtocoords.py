import numpy as np
import cv2 as cv
import os


# if camera is too far away and under weird light conditions, target my split in 2 when shown as contour (check mask2)
 
def nothing(x): pass

def filter_contours_feedback(output, input_contours, min_area, min_perimeter, min_width, max_width, min_height, max_height, min_solidity, max_solidity,
                    min_vertex_count, max_vertex_count, min_ratio, max_ratio):
    passed = []  # returns contours as list of numpy.ndarray
    for contour in input_contours:
        w=9999 #dummy
        h=9999 #dummy
        #print('contour: ', contour)
        center, dim, rot = cv.minAreaRect(contour)
        rot = int(rot)
 
        if rot <= 0 and rot > -45:  # rot=-35,0: flip
            w = dim[0]
            h = dim[1]
        elif (rot >= -90 and rot <= -45):  # rot=-68,-90:good
            w = dim[1]
            h = dim[0]
 
        area = cv.contourArea(contour)
        if (area < min_area):
            if(output==True):
                print("failed area")
            continue
        if (cv.arcLength(contour, True) < min_perimeter):
            if(output==True):
                print("failed perimeter")
            continue
        if (w < min_width or w > max_width):
            if(output==True):
                print("failed width")
            continue
        if (h < min_height or h > max_height):
            if(output==True):
                print("failed height")
            continue
        solid = 100 * area / (0.001+cv.contourArea(cv.convexHull(contour))) #add 0.001 to ensure no division by 0; 0.001 has negligible effect
        if solid < min_solidity or solid > max_solidity:
            if(output==True):
                print("failed solidity")
            continue
        if (len(contour) < min_vertex_count or len(contour) > max_vertex_count):
            if(output==True):
                print("failed vertex_count")
            continue
        ratio = float(w / (h+0.001)) #add 0.001 to ensure no division by 0; 0.001 has negligible effect
        if (ratio < min_ratio or ratio > max_ratio):
            if(output==True):
                print("failed ratio")
            continue
        passed.append(contour)
    return passed
    
    
def order_points_old(pts):
	# initialize a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype="float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis=1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis=1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect

def diyPose(filtered_cnt, rect, img, box):
    cv.drawContours(img, [box], 0, (0, 255, 255), 2)

    M = cv.moments(filtered_cnt[0])
    cx = int(M['m10'] / (M['m00']+0.001)) #add 0.001 to ensure no division by 0; 0.001 has negligible effect
    cy = int(M['m01'] / (M['m00']+0.001)) #add 0.001 to ensure no division by 0; 0.001 has negligible effect

    roll = int(rect[2])

    if roll <= 0 and roll > -45:  # rot=-35,0: flip
        w = rect[1][0]
        h = rect[1][1]
        roll *=-1 
    elif (roll >= -90 and roll <= -45):  # rot=-68,-90:good
        w = rect[1][1]
        h = rect[1][0]
        roll = -90-roll

    cartesian = (cx, 240 - cy)
    distance = (20 * 358.51) / (w+0.00001)  # focal length = 358.51; add 0.00001 to ensure no division by 0; 0.00001 has negligible effect
    # calculate angles
    yaw = (180/3.1415)*np.arctan((w*(cx-160)/20)/distance)
    pitch = (180/3.1415)*np.arctan((h*(cy-120)/14)/distance)
    #print('center:', cartesian, 'distance:', distance, 'width:', w, 'heigth:', h, 'roll:', roll, 'yaw:', yaw, 'pitch:', pitch)    


# capturing video from camera 1 (usb camera) takes 0.094 sec
cap = cv.VideoCapture(0)  

os.system('v4l2-ctl --set-ctrl exposure_auto=1')
os.system('v4l2-ctl --set-ctrl exposure_absolute=5')

param_names = ["min_area", "min_perimeter", "min_width", "max_width", "min_height", "max_height",
                "min_solidity", "max_solidity", "min_vertex_count", "max_vertex_count", "min_ratio", "max_ratio", 
                'minH', "minS", "minV", "maxH", "maxS", "maxV"]
base_params = [0] * 18
filename = "../hsv_params.txt"
file = open(filename, 'r')
params = file.read()

i=0
for word in params.split():
    base_params[i] = int(word)
    i+=1

#creating windows adn trackbars take ~0.1 seconds
cv.namedWindow('Filter bounds')
for i in range(12):
    cv.createTrackbar(param_names[i], 'Filter bounds', base_params[i], 500, nothing)
cv.namedWindow('HSV bounds')
for i in range(6):
    cv.createTrackbar(param_names[i+12], 'HSV bounds', base_params[i+12], 255, nothing) #note that ONLY hue(H) goes 0-180  

n=0
while(1):
    n+=1
    
    for i in range(18): #storing trackbar values in base_params
        if(i<12):
            base_params[i] = cv.getTrackbarPos(param_names[i], 'Filter bounds')
        else:
            base_params[i] = cv.getTrackbarPos(param_names[i], 'HSV bounds')
    
    #0.00014665586 s 

    if cv.waitKey(1) & 0xFF == ord('q'):
        file = open(filename, 'w')
        string = ""
        for i in range(18): #publish trackbar values to file
            string+=(str(base_params[i])+" ")
        params = file.write(string)
        break

    else:        
        e1 = cv.getTickCount() #for benchmarking

        ret, frame = cap.read()  # Capture frame

        resize = cv.resize(frame, None, fx=0.5, fy=0.5)  # resize default(480x640) to 240x320 pixels    

        hsv = cv.cvtColor(resize, cv.COLOR_BGR2HSV)  # Convert BGR to HSV
        lower_hsv = np.array([base_params[12], base_params[13], base_params[14], ])  # lower hsv bounds for filtering
        upper_hsv = np.array([base_params[15], base_params[16], base_params[17], ])  # lower hsv bounds for filtering
        mask = cv.inRange(hsv, lower_hsv, upper_hsv)  # Threshold hsv to only get green values from retroreflective
        cv.imshow("mask", mask)

        mask2 = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((5, 5), np.uint8))  # erode the noise

        #OpenCV 2: findContours returns (contours, hierarchy)
        #OpenCV 3: findContours returns (image, contours, hierarchy)
        #_, cnt, hierarchy = cv.findContours(mask2, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)  # find contours
        cnt, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)  # find contours

        filtered_cnt = filter_contours_feedback(1 , cnt, base_params[0]*2, base_params[1], base_params[2], base_params[3], 
                                        base_params[4], base_params[5], base_params[6]/5, base_params[7]/5, 
                                        base_params[8], base_params[9], base_params[10]/100, base_params[11]/100, )  # filter contours

        #0.03578918491 s
        
        #cv.imshow("resize", resize)
        #cv.imshow("HSV", hsv)
        cv.imshow("mask2", mask2)
        contours = resize.copy() #make copy of resize so resize deosnt have all contours AND filtered contours on same image
        cv.drawContours(contours, cnt, -1, (0, 0, 255), 1)  # draw contours
        #cv.imshow("raw contours", contours)
        filtered_contours = resize.copy()
        cv.drawContours(filtered_contours, filtered_cnt, -1, (255, 255, 0), 1)  # draw contours
        cv.imshow("filtered contours", filtered_contours)

        if (len(filtered_cnt) > 0):
            rect = cv.minAreaRect(filtered_cnt[0])
            box = np.int0(cv.boxPoints(rect))  #In OpenCV 2: cv2.cv.BoxPoints(rect) ; In OpenCV 3: cv.boxPoints(rect)
            ordered_points = order_points_old(box)
            points_list = ordered_points.flatten(order='C')
            string = ",".join(map(str, points_list))


            
            if(n%5 == 0):
                print(n, points_list, string, ordered_points)
                #datafile = open("coords.txt", "a")
                #datafile.write("{},{}\n".format(string, 20))


            
            
        #cv.imshow("with rect", resize)
        print((cv.getTickCount() - e1)/ cv.getTickFrequency()) #for benchmarking
cap.release()  # When everything done, release the capture
cv.destroyAllWindows()  # Close the "Video_Feed" Window
