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

data = sousimage.sousimagemain("predictResults.csv")

for i in range (len(data)):
    coordP, angle = data[i][0], data[i][1]
    long, lat = tt.coord2pixel(coordP)
    heightmax = 35 / tt.distancepixelmoyenne / 10
    heigtmaxopti = 25 / tt.distancepixelmoyenne / 10
    heightminopti = 10 / tt.distancepixelmoyenne / 10 
    tmp = [[0 for _ in range(tt.matrix_width)] for _ in range(tt.matrix_height)]
    tt.draw_circle(tmp, lat, long, round(heightmax))
    tt.addition_matrice(matrix,tmp)
    tmp = [[0 for _ in range(tt.matrix_width)] for _ in range(tt.matrix_height)]
    tt.draw_circle(tmp, lat, long, round(heigtmaxopti))
    tt.addition_matrice(matrix,tmp)
    tmp = [[0 for _ in range(tt.matrix_width)] for _ in range(tt.matrix_height)]
    tt.draw_circle(tmp, lat, long, round(heightminopti))
    tt.soustraction_matrice(matrix,tmp)


print(plotly.__version__, kaleido.__version__)
heatmap_fig = px.imshow(matrix, width=10000, height=5000) #tex_tauto = true for the value in each cell
heatmap_fig.update_traces(dict(showscale=False, 
                       coloraxis=None, 
                       colorscale='rainbow'), selector={'type':'heatmap'})
heatmap_fig.update(layout_coloraxis_showscale=False)
output_file = "heatmap_exportcircle.png"

#affiche heatmap_fig
#heatmap_fig.show() 

pio.write_image(heatmap_fig, output_file)


# End the timer
end_time = time.time()

# Calculate the total execution time
total_time = end_time - start_time

# Print the total execution time
print("Total execution time:", total_time, "seconds")
