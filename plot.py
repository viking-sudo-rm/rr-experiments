import plotly.graph_objects as go
from plotly.subplots import make_subplots


mode = "lines+markers"
lengths = [64, 128, 256, 512, 1024, 2048]
size = 500
font_size = 25

lstm = "LSTM"
qrnn = "QRNN"
qrnn_ex = "QRNN+"
qrnn2 = "QRNN2"
always_true = "Always True"
always_false = "Always False"
baseline = "Majority Class"

# Results for a^nb^n.
anbn = [
    (lstm, [100, 100, 99.58, 97.85, 94.11, 89.47]),
    (qrnn, [90.23, 78.31, 67.01, 61.32, 58.61, 57.68]),
    (qrnn_ex, [99.60, 87.33, 81.53, 82.11, 84.08, 86.97]),
    (qrnn2, [100, 99.3, 90.7, 65.75, 58.11, 59.22]),
    (baseline, [37.52, 52.31, 64.17, 73.22, 79.73, 84.73]),
]

# Results for a^nb^nw.
anbnw = [
    (lstm, [100, 100, 99.00, 95.54, 92.73, 88.64]),
    (qrnn, [79.95, 82.00, 79.04, 80.37, 79.52, 76.52]),
    (qrnn_ex, [82.01, 82.65, 78.82, 80.24, 77.82, 76.26]),
    (qrnn2, [100, 98.05, 91.62, 89.60, 84.91, 78.67]),
    (baseline, [79.75, 81.86, 78.78, 80.25, 79.24, 76.37]),
    # (always_false, [20.25, 18.14, 21.22, 19.75, 20.76, 23.63]),
]

data = anbnw

# Default plotly colors.
colors = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut cleawn
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'   # blue-teal
]

fig = go.Figure()
fig = make_subplots(rows=2, cols=1,
                    shared_xaxes=True,
                    horizontal_spacing = 0.03,
                    vertical_spacing = 0.03,
                    subplot_titles=["$L_5$<br>", "$a^nb^nw$<br>"])

for idx, (name, y) in enumerate(anbn):
    trace = go.Scatter(x=lengths, y=y,
                       mode=mode,
                       name=name,
                       line={"color": colors[idx]})
    fig.add_trace(trace, row=1, col=1)

for idx, (name, y) in enumerate(anbnw):
    trace = go.Scatter(x=lengths, y=y,
                       mode=mode,
                       name=name,
                       line={"color": colors[idx]},
                       showlegend=False)
    fig.add_trace(trace, row=2, col=1)


fig.update_xaxes(type="log")
fig.update_xaxes(title_text=None, row=1, col=1)
fig.update_xaxes(title_text="<i>Length</i>", row=2, col=1)
fig.update_yaxes(title_text="<i>Accuracy</i>")

fig.update_layout(legend_orientation="h",
                  width=1.6 * size,
                  height=2 * size,
                  legend=dict(
                    xanchor="center",
                    yanchor="top",
                    y=-.13,
                    x=.5,
                    font=dict(family="Times New Roman", size=25),
                  ),
                  font=dict(family="Times New Roman", size=font_size),
                  # title={

                  #   "text": '<b>Generalization Accuracy by Length</b>',
                  #   "x": .5,
                  #   "font": dict(family="Times New Roman", size=35),
                  # }
                  )

for i in fig['layout']['annotations']:
    i['font'] = dict(
        family="Times New Roman",
        size=font_size)

fig.show()
