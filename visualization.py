import os
import pandas as pd
import matplotlib.pyplot as plt

# Plots IMU data for 3 sensors, xyz each. Target label of freezing or non-freezing is overlayed.
def x_plot(file):
    data = pd.read_csv(file, delim_whitespace=True, header = None)
    type_column = list(data[10])
    pcol = ['w','g','r']
    yltext=['X','Y','Z']
    ttext=['sensor ankle','sensor knee', 'sensor hip']

    fig=plt.figure(figsize=(20,16))
    for sensorpos in range(3):
      for sensoraxis in range(3):
        ax = fig.add_subplot(3, 3, 1+sensorpos + 3*sensoraxis)
        switch = [i for i in range(1,len(type_column)) if type_column[i]!=type_column[i-1]]
        switch = [0] + switch + [len(type_column)-1]
        for i in range(len(switch)-1):
          x1 = (data[0][switch[i]])/1000
          x2 = (data[0][switch[i+1]])/1000
          data_type = type_column[switch[i]+1]
          ax.axvspan(x1, x2, 0, 0.077, facecolor=pcol[data_type], alpha=1.0, edgecolor='black', zorder=1)
        ax.plot(data[0].values/1000, data[1+sensoraxis+3*sensorpos].values, linewidth=0.2, zorder=2)

        ax.set_xlim([0, data[0].iloc[-1]/1000])
        ax.set_ylim([-3500, 3000])
        ax.set_xlabel('time [s]')
        ax.set_ylabel('Acc {}[mg]'.format(yltext[sensoraxis]))
        ax.set_title(ttext[sensorpos])

    plot_title = file.split('/')[-1].split('.')[0]
    plt.show()
    plt.savefig(plot_title + '.png', format='PNG', dpi=300)
