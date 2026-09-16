'''
Generates a basic Sankey diagram using Plotly.
'''


import plotly.graph_objects as go

labels = ["Source A", "Source B", "Source C", "Destination A", "Destination B", "Destination C"]

source = [0, 1, 2, 0, 1, 2]
target = [3, 3, 3, 4, 4, 4]
value = [8, 4, 2, 8, 2, 4]

fig = go.Figure(data=[go.Sankey(
    node = dict(
      pad = 15,
      thickness = 20,
      line = dict(color = "black", width = 0.5),
      label = labels
    ),
    link = dict(
      source = source,
      target = target,
      value = value
  ))])

fig.update_layout(title_text="Basic Sankey Diagram", font_size=10)
fig.show()

