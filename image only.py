import sys

sys.path.append('/usr/local/lib/python2.7/site-packages')

import cv2
import numpy as np
import utlis
###############################
webCamFeed = True
cap = cv2.VideoCapture(0)
cap.set(10,160)
heightImg = 800
widthImg  = 800
questions=10
choices=5
pathImage='z.jpg'
ans=[0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3]
count=0
   
img = cv2.imread(pathImage)
img = cv2.resize(img, (widthImg, heightImg)) # RESIZE IMAGE
imgFinal = img.copy()
imgBlank = np.zeros((heightImg,widthImg, 3), np.uint8) # CREATE A BLANK IMAGE FOR TESTING DEBUGGING IF REQUIRED
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # CONVERT IMAGE TO GRAY SCALE
imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1) # ADD GAUSSIAN BLUR
imgCanny = cv2.Canny(imgBlur,10,70) # APPLY CANNY 
try:
    ## FIND ALL COUNTOURS
    imgContours = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
    imgBigContour = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
    contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # FIND ALL CONTOURS
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10) # DRAW ALL DETECTED CONTOURS
    rectCon = utlis.rectContour(contours) # FILTER FOR RECTANGLE CONTOURS
    biggestPoints= utlis.getCornerPoints(rectCon[0]) # GET CORNER POINTS OF THE BIGGEST RECTANGLE
    gradePoints = utlis.getCornerPoints(rectCon[1]) # GET CORNER POINTS OF THE SECOND BIGGEST RECTANGLE
    if biggestPoints.size != 0 and gradePoints.size != 0:
        sorted_rectCon_area = sorted(rectCon, key=cv2.contourArea, reverse=True)

        # Extract SBD (biggest) and Textcode (smallest)
        #SBD = sorted_rectCon_area[0]
        #Textcode = sorted_rectCon_area[-1]

        # Remove SBD and Textcode from the list
        rectCon = sorted_rectCon_area[1:]

        # Sort the rectCon
        center_points = []
        for rect in rectCon:
            points = utlis.getCornerPoints(rect)
            center_x = int((points[0][0][0] + points[1][0][0] + points[2][0][0] + points[3][0][0])/4)
            center_y = int((points[0][0][1] + points[1][0][1] + points[2][0][1] + points[3][0][1])/4)
            center_points.append((center_x, center_y))

        # Normalize center points
        def normalize(points):
            normalized_points = []
            for x, y in points:
                normalized_x = round(x / 10) * 10
                normalized_y = round(y / 10) * 10
                normalized_points.append((normalized_x, normalized_y))
            return normalized_points

        normalized_center_points = normalize(center_points)

        # Sort rectCon based on normalized center points
        sorted_rectCon = [rect for _, rect in sorted(zip(normalized_center_points, rectCon), key=lambda item: (item[0][1], item[0][0]))]


        # Initialize imgFinal_masked outside the loop
        imgFinal_masked = imgFinal.copy()
        useranswers=[]
        n = len(rectCon)  # Number of subsets

        k = 10  # Number of elements per subset

        ans_subsets=[]
        k=10
        for i in range (0, len(ans),k):
            temp = ans[i:i + k]
            ans_subsets.append(temp)

        Totalscore = 0
        for j in range(min(len(ans_subsets), len(rectCon))):
            i = sorted_rectCon[j]
            biggestPoints= utlis.getCornerPoints(i) # GET CORNER POINTS OF THE BIGGEST RECTANGLE
            biggestPoints=utlis.reorder(biggestPoints) # REORDER FOR WARPING
            cv2.drawContours(imgBigContour, biggestPoints, -1, (0, 255, 0), 20) # DRAW THE BIGGEST CONTOUR
            pts1 = np.float32(biggestPoints) # PREPARE POINTS FOR WARP
            pts2 = np.float32([[0, 0],[800, 0], [0, 1600],[800, 1600]]) # PREPARE POINTS FOR WARP
            matrix = cv2.getPerspectiveTransform(pts1, pts2) # GET TRANSFORMATION MATRIX
            imgWarpColored = cv2.warpPerspective(img, matrix, (800, 1600)) # APPLY WARP PERSPECTIVE
            imgWarpGray = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY) # CONVERT TO GRAYSCALE
            imgWarpGrayContrast = cv2.convertScaleAbs(imgWarpGray, alpha=1.5, beta=0)
            imgThresh = cv2.threshold(imgWarpGray, 120, 255,cv2.THRESH_BINARY_INV )[1] # APPLY THRESHOLD AND INVERSE
            cv2.imshow("imgThresh", imgThresh)
            boxes = utlis.splitBoxes(imgThresh, questions, choices) # GET INDIVIDUAL BOXES
            countR=0
            countC=0
            myPixelVal = np.zeros((questions,choices)) # TO STORE THE NON ZERO VALUES OF EACH BOX
            myIndex = []
            for image in boxes:
                #cv2.imshow(str(countR)+str(countC),image)
                totalPixels = cv2.countNonZero(image)
                myPixelVal[countR][countC]= totalPixels
                countC += 1
                if (countC==choices):countC=0;countR +=1

            # FIND THE USER ANSWERS AND PUT THEM IN A LIST
            for x in range (0,questions):
                arr = myPixelVal[x]
                myIndexVal = np.where(arr == np.amax(arr))
                myIndex.append(myIndexVal[0][0])
            print("USER ANSWERS",myIndex)
            useranswers.append(myIndex)
            flat_useranswers = [item for sublist in useranswers for item in sublist]
            print("FLAT USER ANSWERS",flat_useranswers)
            # CALCULATE THE GRADING
            grading=[]

            for x in range(0,questions):
                if ans_subsets[j][x] == myIndex[x]:
                    grading.append(1)
                else:grading.append(0)
                        #print("GRADING",grading)
            score = (sum(grading)/questions)*100 # FINAL GRADE
            print("SCORE",score)
            Totalscore += score
            print("TOTALSCORE",Totalscore)

            utlis.showAnswers(imgWarpColored,myIndex,grading,ans_subsets[j],questions,choices) # DRAW DETECTED ANSWERS
            utlis.drawGrid(imgWarpColored,questions, choices) # DRAW GRID
            imgRawDrawings = np.zeros_like(imgWarpColored) # NEW BLANK IMAGE WITH WARP IMAGE SIZE
            utlis.showAnswers(imgRawDrawings, myIndex,grading,ans_subsets[j],questions,choices) # DRAW ON NEW IMAGE
            invMatrix = cv2.getPerspectiveTransform(pts2, pts1) # INVERSE TRANSFORMATION MATRIX
            imgInvWarp = cv2.warpPerspective(imgRawDrawings, invMatrix, (widthImg, heightImg)) # INV IMAGE WARP
            #cv2.imshow(imgInvWarp)

            # Create a mask for imgInvWarp
            imgInvWarp_gray = cv2.cvtColor(imgInvWarp, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(imgInvWarp_gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)

            # Black out the area of imgFinal where imgInvWarp will be placed
            imgFinal_masked = cv2.bitwise_and(imgFinal_masked, imgFinal_masked, mask=mask_inv)
            #cv2.imshow(imgFinal_masked)
            # Place imgInvWarp on top of the masked imgFinal
            imgFinal_overlayed = cv2.add(imgFinal_masked, imgInvWarp)


         
            # Update imgFinal_masked for the next iteration
            imgFinal_masked = imgFinal_overlayed.copy()
        Totalscore=Totalscore/len(ans)
        print("Final_TOTALSCORE",Totalscore)    
        cv2.imshow("Final Result", imgFinal_overlayed)
        # Export the final result to a separate file
        cv2.imwrite("final_result.jpg", imgFinal_overlayed)
        cv2.waitKey(0)
    
except:
    imageArray = ([img,imgGray,imgCanny,imgContours],
                        [imgBlank, imgBlank, imgBlank, imgBlank])