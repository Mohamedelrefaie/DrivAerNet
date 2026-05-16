import numpy as np
import matplotlib.pyplot as plt

#!----------------
    fig = plt.figure(figsize=(10, 5))
    -> Create a new figure for plotting
    -> set width 10 inches, height to 5 inches

#!----------------
    ax = fig.add_subplot(121, projection='3d')
    -> Add a subplot to your figure
    -> 121
       -> 1 row, 2 columns
       -> "1st" The first subplot
    -> projection='3d'
       -> Create a "3D" plot

#!----------------
    p = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=true_p, cmap='jet', s=1)
    -> Create a "3D" scatter plot on the first subplot "ax"
    -> points[:,0]
       -> x-coordinate of each point
    -> points[:,1]
       -> y-coordinate of each point
    -> points[:,2]
       -> z-coordinate of each point
    -> c=trup_p
       -> Color each point using "true pressure value"
    -> cmap='jet'
       -> Use jet color map(blue -> green -> yellow -> red)
       -> Low  values: blue
       -> Mid  values: yellow
       -> High values: red
    -> s=1
       -> s stands for 'size' of the points in the scatter plot
       -> s = 10
          -> Medium points
       -> s = 50
          -> Big points

#!----------------
    fig.colorbar(p, ax=ax)
    -> Add a colorbar next to the first subplot

#!----------------
    os.makedirs(visualization_path, exist_ok=True)
    -> Create folder if it does not exist

#!----------------
    plt.savefig(os.path.join(visualization_path, "visualization.png"), dpi=300)
    -> dpi
       -> dots for per inch
       -> low resolution
          -> dpi = 72
          -> common for screen display
       -> high resolution
          -> dpi = 300
          -> Used in scientific paper

#!----------------
    plt.tight_layout()
    -> Automatically adjusts the spacing between subplots and surrounding text to the prevent overlap




