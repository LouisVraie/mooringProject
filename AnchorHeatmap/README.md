convcoordcirle.py is for generating the heatmap with circle that have a confidence interval
convcoordtriangle.py is for the heatmap with triangle
convcoordtrianglezoning is for the heatmap with triangle and the confidence interval

BaseImage have the calibration picture used for the code
You can change the calibration of the coordinate of the corner in line 155 of forme.py

Orientation.py is for detecting the orientation of a picture
forme.py is the multiple tools used for the graphical part
sousimage.py is for creating the subimage and send to orientation.py to be processed. The outputed picture can be found in Orientation/Input

predictResults.csv is the data that the program will use.

For reducing the amount of file on the git, they is not all the picture, the arborescence need to be the same as the image field in the csv

Credits : Laurent Peraldi
