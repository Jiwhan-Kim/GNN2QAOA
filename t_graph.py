from graphs import create_long_graph
import matplotlib.pyplot as plt
from rustworkx.visualization import mpl_draw

graph, edge_list = create_long_graph(15)

# max_str = "010101010101010"
max_str = "101010101010101"

node_color = ['lightblue' if bit == '0'
              else 'orange' for bit in max_str]

plt.figure(5)
mpl_draw(graph, with_labels=True,
         node_color=node_color, font_size=15)
plt.savefig(f"./images/qqq2-5-result.png")
