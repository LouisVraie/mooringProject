import plotly.express as px
import pandas as pd
import forme as tt
import plotly.io as pio
import kaleido
import plotly
import time
import sousimage as sousimage

# Start the timer
start_time = time.time()

#Matrice de la heatmap
matrix = [[0 for _ in range(tt.matrix_width)] for _ in range(tt.matrix_height)]

#TEST port honorat :  43°30'33.98"N   7° 2'49.54"E
#TEST port honorat :  43.5094361  7.0470861
#expected result : 2662, 2353 px
#port = (tt.coord(43, 30, 33.97), tt.coord(7, 2, 49.51))
#print("testfunction",tt.coord2pixel(port))

#matrix[int(y/10)][int(x/10)] = 1

data = sousimage.sousimagemain("predictResults.csv")


for i in range (len(data)):
    coordP, angle = data[i][0], data[i][1]
    long, lat = tt.coord2pixel(coordP)
    height = 35 / tt.distancepixelmoyenne / 10
    x1,y1,x2,y2 = tt.find_base_points(long,lat,height, angle)
    tmp = [[0 for _ in range(tt.matrix_width)] for _ in range(tt.matrix_height)]
    tt.draw_triangle(tmp, x1, y1,  x2, y2,  long, lat)
    tt.addition_matrice(matrix,tmp)


print(plotly.__version__, kaleido.__version__)
heatmap_fig = px.imshow(matrix, width=10000, height=5000)
#heatmap_fig = px.imshow(matrix, text_auto=True)
heatmap_fig.update_traces(dict(showscale=False, 
                       coloraxis=None, 
                       colorscale='rainbow'), selector={'type':'heatmap'})
heatmap_fig.update(layout_coloraxis_showscale=False)
output_file = "heatmap_exporttriangle.png"

#affiche heatmap_fig
#heatmap_fig.show()

pio.write_image(heatmap_fig, output_file)


# End the timer
end_time = time.time()

# Calculate the total execution time
total_time = end_time - start_time

# Print the total execution time
print("Total execution time:", total_time, "seconds")
