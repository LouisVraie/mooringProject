import cv2
import numpy as np


# load image as HSV and select saturation
def trouver_angle(image):
    liste = []
    img = cv2.imread(image)
    hh, ww, cc = img.shape

    # convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # threshold the grayscale image   bateau : 160
    for j in range (0,250,10):
        ret, thresh = cv2.threshold(gray,j, 255, cv2.THRESH_BINARY)
        for i in range (20):
        # Définir le noyau de l'élément structurant (kernel)
            kernel = np.ones((i, i), np.uint8) 

            # Appliquer une fermeture à l'image
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            # find outer contour 
            cntrs = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]

            # get rotated rectangle from outer contour
        
            if len(cntrs) != 0:
                
                rotrect = cv2.minAreaRect(cntrs[0])
                box = cv2.boxPoints(rotrect)
                box = np.int0(box)

                # draw rotated rectangle on copy of img as result
                result = img.copy()
                
                cv2.drawContours(result,[box],0,(0,0,255),2)

                # get angle from rotated rectangle
                angle = rotrect[-1]

                
                if (angle % 1 != 0):
                    liste.append(angle)
                    # write result to disk
                    #cv2.imwrite("Orientation\\output\\"+image+"result.png", result)
                    
                    """cv2.imshow("THRESH", thresh)
                    cv2.imshow("RESULT", result)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()"""
            else :
                liste.append(0)
    angle_moyen = sum(liste)/len(liste)
    return angle_moyen