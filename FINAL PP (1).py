# -*- coding: utf-8 -*-
"""
Created on Tues Nov  1 15:58:26 2022

@author: omar.al-khateeb
"""

############ IMP0RT LASI0 FILE ############

import lasio as las      #pip install lasio

data = las.read('1051661435.las')
data = data.df()

############ CLEANING DATA ############

import numpy as np
import missingno as msno      #pip install missingno

msno.matrix(data)

data['CNDL'][data['CNDL'] < 0] = np.nan
data['CNDL'][data['CNDL'] > 70] = np.nan

data['CNLS'][data['CNLS'] < 0] = np.nan

data['CNPOR'][data['CNPOR'] < 0] = np.nan

data['GR'][data['GR'] > 500] = np.nan
data['GR'][data['GR'] < 0] = np.nan

data['RILD'][data['RILD'] > 2500] = np.nan

data['RILM'][data['RILM'] > 8500] = np.nan

data['MCAL'][data['MCAL'] < 0] = np.nan
data['MI'][data['MI'] < 0] = np.nan
data['MN'][data['MN'] < 0] = np.nan
data['DT'][data['DT'] < 0] = np.nan

data['SPOR'][data['SPOR'] < 0] = np.nan

data['RHOB'][data['RHOB'] < 0] = np.nan
data['RHOB'][data['RHOB'] > 5] = np.nan

data['RHOC'][data['RHOC'] <= 0] = np.nan

data['DPOR'][data['DPOR'] < 0] = np.nan
data['DPOR'][data['DPOR'] > 90] = np.nan


data.index.names = ['DEPTH']
data['DEPTH'] = data.index
msno.matrix(data)

data = data.dropna()
msno.matrix(data)


############ VISUALIZAING DATA ############

import matplotlib.pyplot as plt

data.plot(x='GR', y='DEPTH', c='black', lw=0.5, legend=False, figsize=(7,10))
plt.ylim(8764/2, 7000/2)
plt.xlim(0,150)
plt.title('Plot With a Single Colour Fill to Y-Axis')
plt.fill_betweenx(data['DEPTH'], data['GR'], 0, facecolor='green')
plt.fill_betweenx(data['DEPTH'], data['GR'], 150, facecolor='yellow')
plt.show()
print('Thickness of reservoir is', (data['GR'] < 40).sum()*0.5, 'ft')

## [OUT] : 403.0ft

fig_res, axes = plt.subplots(1,3, sharex=True, sharey=True, figsize=(21,10))
data.plot(x='RILD', y='DEPTH', ax=axes[0], legend=False, logx=True, lw=0.5, c='red')
data.plot(x='RILM', y='DEPTH',ax=axes[1], legend=False, logx=True, lw=0.5, c ='blue')
data.plot(x='RLL3', y='DEPTH',ax=axes[2], legend=False, logx=True, lw=0.5, c='black')
axes[0].set_xlim(0,500)
axes[0].set_ylim(8211/2, 7000/2)
fig_res.suptitle('Resistivity (ohmmeter)', fontsize='x-large')
axes[0].set_ylabel('Depth (ft)')

print('Thickness of reservoir is', (data.loc[(data['GR'] < 40),'RILM'] > 15).sum()*0.5, 'ft')

## [OUT] : 118.0ft

fig = plt.subplots(figsize=(7,10))

ax1 = plt.subplot2grid((1,1), (0,0), rowspan=1, colspan=1)
ax2 = ax1.twiny()

ax1.plot('RHOB', 'DEPTH', data=data, color='red', lw=0.5)
ax1.set_xlim(1.95, 2.95)
ax1.set_ylim(8211/2, 7000/2)
ax1.set_xlabel('Density')
ax1.xaxis.label.set_color("red")
ax1.tick_params(axis='x', colors="red")
ax1.spines["top"].set_edgecolor("red")

ax2.plot('CNPOR', 'DEPTH', data=data, color='blue', lw=0.5)
ax2.set_xlim(45, -15)
ax2.set_ylim(8211/2, 7000/2)
ax2.set_xlabel('Neutron')
ax2.xaxis.label.set_color("blue")
ax2.spines["top"].set_position(("axes", 1.08))
ax2.tick_params(axis='x', colors="blue")
ax2.spines["top"].set_edgecolor("blue")

x1=data['RHOB']
x2=data['CNPOR']

x = np.array(ax1.get_xlim())
z = np.array(ax2.get_xlim())

nz=((x2-np.max(z))/(np.min(z)-np.max(z)))*(np.max(x)-np.min(x))+np.min(x)

ax1.fill_betweenx(data['DEPTH'], x1, nz, where=x1>=nz, interpolate=True, color='green')
ax1.fill_betweenx(data['DEPTH'], x1, nz, where=x1<=nz, interpolate=True, color='yellow')

for ax in [ax1, ax2]:
    ax.set_ylim(8211/2, 7000/2)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")




########### ANOTHER WAY TO VISUALIZE DATA ################


def well_log_display(df, column_depth, column_list, 
                     column_semilog=None, min_depth=None, max_depth=None, 
                     column_min=None, column_max=None, colors=None, 
                     fm_tops=None, fm_depths=None, 
                     tight_layout=1, title_size=10):
  """
  Display log side-by-side style
  Input:
  df is your dataframe
  specify min_depth and max_depth as the upper and lower depth limit
  column_depth is the column name of your depth
  column_list is the LIST of column names that you will display
  column_semilog is specific for resistivity column; if your resistivities are 
    in column 3, specify as: column_semilog=2. Default is None, so if you don't 
    specify, the resistivity will be plotted in normal axis instead
    
  column_min is list of minimum values for the x-axes.
  column_max is list of maximum values for the x-axes.
  
  colors is the list of colors specified for each log names. Default is None,
    so if don't specify, the colors will be Matplotlib default (blue)
  fm_tops and fm_depths are the list of formation top names and depths.
    Default is None, so no tops are shown. Specify both lists, if you want
    to show the tops
  """
  import numpy as np
  import matplotlib.pyplot as plt
  import pandas as pd
  import random

  if column_semilog==None:
    # column semilog not defined, RT will be plotted in normal axis
    logs = column_list

    # create the subplots; ncols equals the number of logs
    fig, ax = plt.subplots(nrows=1, ncols=len(logs), figsize=(20,10))

    # looping each log to display in the subplots
    if colors==None:
      # color is None (default)
      for i in range(len(logs)):
        # normal axis plot
        ax[i].plot(df[logs[i]], df[column_depth])
        ax[i].set_title(logs[i], size=title_size)
        ax[i].minorticks_on()
        ax[i].grid(which='major', linestyle='-', linewidth='0.5', color='lime')
        ax[i].grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        if column_min!=None and column_max!=None:
          # x-axis limits defined
          ax[i].set_xlim(column_min[i], column_max[i])
        if min_depth!=None and max_depth!=None:
          # y-axis limit defined
          ax[i].set_ylim(min_depth, max_depth)          
        ax[i].invert_yaxis()    

    else:
      # colors are defined (as list)
      for i in range(len(logs)):
        # normal axis plot
        ax[i].plot(df[logs[i]], df[column_depth], color=colors[i])
        ax[i].set_title(logs[i], size=title_size)
        ax[i].minorticks_on()
        ax[i].grid(which='major', linestyle='-', linewidth='0.5', color='lime')
        ax[i].grid(which='minor', linestyle=':', linewidth='0.5', color='black')        
        if column_min!=None and column_max!=None:
          # x-axis limits defined
          ax[i].set_xlim(column_min[i], column_max[i])       
        if min_depth!=None and max_depth!=None:
          # y-axis limit defined
          ax[i].set_ylim(min_depth, max_depth)           
        ax[i].invert_yaxis()    


  else:
    # column semilog is defined, RT will be plotted in semilog axis
    logs = column_list

    # create the subplots; ncols equals the number of logs
    fig, ax = plt.subplots(nrows=1, ncols=len(logs), figsize=(20,10))

    # looping each log to display in the subplots
    if colors==None:
      # color is None (default)
      for i in range(len(logs)):
        if i == column_semilog:
          # for resistivity, semilog plot
          ax[i].semilogx(df[logs[i]], df[column_depth])
        else:
          # for non-resistivity, normal plot
          ax[i].plot(df[logs[i]], df[column_depth])
        
        ax[i].set_title(logs[i], size=title_size)
        ax[i].minorticks_on()
        ax[i].grid(which='major', linestyle='-', linewidth='0.5', color='lime')
        ax[i].grid(which='minor', linestyle=':', linewidth='0.5', color='black')        
        if column_min!=None and column_max!=None:
          # x-axis limits defined
          ax[i].set_xlim(column_min[i], column_max[i])        
        if min_depth!=None and max_depth!=None:
          # y-axis limit defined
          ax[i].set_ylim(min_depth, max_depth)          
        ax[i].invert_yaxis()    

    else:
      # colors are defined (as list)
      for i in range(len(logs)):
        if i == column_semilog:
          # for resistivity, semilog plot
          ax[i].semilogx(df[logs[i]], df[column_depth], color=colors[i])     
        else:
          # for non-resistivity, normal plot
          ax[i].plot(df[logs[i]], df[column_depth], color=colors[i])
        
        ax[i].set_title(logs[i], size=title_size)
        ax[i].minorticks_on()
        ax[i].grid(which='major', linestyle='-', linewidth='0.5', color='lime')
        ax[i].grid(which='minor', linestyle=':', linewidth='0.5', color='black')  
        if column_min!=None and column_max!=None:
          # x-axis limits defined
          ax[i].set_xlim(column_min[i], column_max[i])   
        if min_depth!=None and max_depth!=None:
          # y-axis limit defined
          ax[i].set_ylim(min_depth, max_depth)
        ax[i].invert_yaxis() 

  if fm_tops!=None and fm_depths!=None:
    # Formation tops and depths are specified, they will be shown

    # produce colors
    rgb = []
    for j in range(len(fm_tops)):
      _ = (random.random(), random.random(), random.random())
      rgb.append(_)

    for i in range(len(logs)):
      for j in range(len(fm_tops)):
        # rgb = (random.random(), random.random(), random.random())
        ax[i].axhline(y=fm_depths[j], linestyle=":", c=rgb[j], label=fm_tops[j])  
        # y = fm_depths[j] / (max_depth - min_depth)    
        # ax[i].text(0.5, y, fm_tops[j], fontsize=5, va='center', ha='center', backgroundcolor='w')

  # plt.legend()
  # plt.legend(loc='upper center', bbox_to_anchor=(-3, -0.05),
  #            fancybox=True, shadow=True, ncol=5)  
  
  plt.show() 


df = data
column_depth = 'DEPTH'
column_list = ['RHOB', 'DT', 'GR', 'CNPOR', 'RILM']
column_semilog = 6
column_min=None
column_max=None
min_depth= 3500
max_depth= 4150
colors=["black","blue", "orange", "red","grey","green","yellow"]
well_log_display(df, column_depth, column_list, column_semilog, min_depth, max_depth, column_min, column_max, colors)




def triple_combo(df, column_depth, column_GR, column_resistivity, 
                 column_NPHI, column_RHOB, min_depth, max_depth, 
                 min_GR=0, max_GR=150, sand_GR_line=60,
                 min_resistivity=0.01, max_resistivity=1000, 
                 color_GR='black', color_resistivity='green', 
                 color_RHOB='red', color_NPHI='blue',
                 figsize=(6,10), tight_layout=1, 
                 title_size=15, title_height=1.05):
  """
  Producing Triple Combo log
  Input:
  df is your dataframe
  column_depth, column_GR, column_resistivity, column_NPHI, column_RHOB
  are column names that appear in your dataframe (originally from the LAS file)
  specify your depth limits; min_depth and max_depth
  input variables other than above are default. You can specify
  the values yourselves. 
  Output:
  Fill colors; gold (sand), lime green (non-sand), blue (water-zone), orange (HC-zone)
  """
  
  import matplotlib.pyplot as plt
  from matplotlib.ticker import AutoMinorLocator  
  import numpy as np

  fig, ax=plt.subplots(1,3,figsize=(8,10))
  fig.suptitle('Triple Combo Log', size=title_size, y=title_height)

  ax[0].minorticks_on()
  ax[0].grid(which='major', linestyle='-', linewidth='0.5', color='lime')
  ax[0].grid(which='minor', linestyle=':', linewidth='1', color='black')

  ax[1].minorticks_on()
  ax[1].grid(which='major', linestyle='-', linewidth='0.5', color='lime')
  ax[1].grid(which='minor', linestyle=':', linewidth='1', color='black')

  ax[2].minorticks_on()
  ax[2].grid(which='major', linestyle='-', linewidth='0.5', color='lime')
  ax[2].grid(which='minor', linestyle=':', linewidth='1', color='black')  

  # First track: GR
  ax[0].get_xaxis().set_visible(False)
  ax[0].invert_yaxis()   

  gr=ax[0].twiny()
  gr.set_xlim(min_GR,max_GR)
  gr.set_xlabel('GR',color=color_GR)
  gr.set_ylim(max_depth, min_depth)
  gr.spines['top'].set_position(('outward',10))
  gr.tick_params(axis='x',colors=color_GR)
  gr.plot(df[column_GR], df[column_depth], color=color_GR)  

  gr.minorticks_on()
  gr.xaxis.grid(which='major', linestyle='-', linewidth='0.5', color='lime')
  gr.xaxis.grid(which='minor', linestyle=':', linewidth='1', color='black') 

  gr.fill_betweenx(df[column_depth], sand_GR_line, df[column_GR], where=(sand_GR_line>=df[column_GR]), color = 'gold', linewidth=0) # sand
  gr.fill_betweenx(df[column_depth], sand_GR_line, df[column_GR], where=(sand_GR_line<df[column_GR]), color = 'lime', linewidth=0) # shale

  # Second track: Resistivity
  ax[1].get_xaxis().set_visible(False)
  ax[1].invert_yaxis()   

  res=ax[1].twiny()
  res.set_xlim(min_resistivity,max_resistivity)
  res.set_xlabel('Resistivity',color=color_resistivity)
  res.set_ylim(max_depth, min_depth)
  res.spines['top'].set_position(('outward',10))
  res.tick_params(axis='x',colors=color_resistivity)
  res.semilogx(df[column_resistivity], df[column_depth], color=color_resistivity)    

  res.minorticks_on()
  res.xaxis.grid(which='major', linestyle='-', linewidth='0.5', color='lime')
  res.xaxis.grid(which='minor', linestyle=':', linewidth='1', color='black')   

  # Third track: NPHI and RHOB
  ax[2].get_xaxis().set_visible(False)
  ax[2].invert_yaxis()  

  ## NPHI curve 
  nphi=ax[2].twiny()
  nphi.set_xlim(-0.15,0.45)
  nphi.invert_xaxis()
  nphi.set_xlabel('NPHI',color='blue')
  nphi.set_ylim(max_depth, min_depth)
  nphi.spines['top'].set_position(('outward',10))
  nphi.tick_params(axis='x',colors='blue')
  nphi.plot(df[column_NPHI], df[column_depth], color=color_NPHI)

  nphi.minorticks_on()
  nphi.xaxis.grid(which='major', linestyle='-', linewidth='0.5', color='lime')
  nphi.xaxis.grid(which='minor', linestyle=':', linewidth='1', color='black')     

  ## RHOB curve 
  rhob=ax[2].twiny()
  rhob.set_xlim(1.95,2.95)
  rhob.set_xlabel('RHOB',color='red')
  rhob.set_ylim(max_depth, min_depth)
  rhob.spines['top'].set_position(('outward',50))
  rhob.tick_params(axis='x',colors='red')
  rhob.plot(df[column_RHOB], df[column_depth], color=color_RHOB)

  # solution to produce fill between can be found here:
  # https://stackoverflow.com/questions/57766457/how-to-plot-fill-betweenx-to-fill-the-area-between-y1-and-y2-with-different-scal
  x2p, _ = (rhob.transData + nphi.transData.inverted()).transform(np.c_[df[column_RHOB], df[column_depth]]).T
  nphi.autoscale(False)
  nphi.fill_betweenx(df[column_depth], df[column_NPHI], x2p, color="orange", alpha=0.4, where=(x2p > df[column_NPHI])) # hydrocarbon
  nphi.fill_betweenx(df[column_depth], df[column_NPHI], x2p, color="blue", alpha=0.4, where=(x2p < df[column_NPHI])) # water

  res.minorticks_on()
  res.grid(which='major', linestyle='-', linewidth='0.5', color='lime')
  res.grid(which='minor', linestyle=':', linewidth='1', color='black')
    
  plt.show()



df = data
column_depth = 'DEPTH'
column_GR = 'GR'
column_resistivity = 'RILM'
column_NPHI = 'CNPOR'
column_RHOB  = 'RHOB'
min_depth= 3500
max_depth= 4200
triple_combo(df, column_depth, column_GR, column_resistivity, column_NPHI, column_RHOB, min_depth, max_depth)

## EXTRA



Density = data['RHOB']
Density = Density.to_numpy()

Porosity = data['CNPOR']
Porosity = Porosity.to_numpy()

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(Porosity, Density, color = 'orange', edgecolors = 'brown')
plt.xlabel("Porosity (%)")
plt.ylabel("Bulk Density (g/cc)")
plt.show()



############ REGRESSION ############ 

import matplotlib.pyplot as plt
import pandas as pd

poro = pd.read_csv('poro_perm_data.csv')
poro = poro.dropna()

dataclean = (poro["Permeability (mD)"] > 0)  & (poro['Porosity (%)'] > 0) 
poro = poro[dataclean]


poroplot = poro.plot('Depth (ft)','Porosity (%)', kind = 'scatter')
permplot = poro.plot('Depth (ft)','Permeability (mD)',kind = 'scatter')


poro['Porosity (%)'].plot(kind='hist')
plt.ylabel('Frequency')
plt.xlabel('Porosity (%)')
plt.show()

poro['Permeability (mD)'].plot(kind='hist')
plt.ylabel('Frequency')
plt.xlabel('Permeability (mD)')
plt.show()

from sklearn.linear_model import LinearRegression

porox=poro.loc[:,'Porosity (%)'].to_numpy().reshape(-1,1)
permy=poro.loc[:,'Permeability (mD)'].to_numpy().reshape(-1,1)
fig_reg, ax = plt.subplots()
ax.scatter(porox,permy)
ax.set_xlabel('Porosity (%)')
ax.set_ylabel('Permeability (mD)')
ax.set_title('Porosity vs. Permeability')
LR = LinearRegression()
LR.fit(porox, permy)
r_sq = LR.score(porox, permy)
y_pred = LR.predict(porox)
ax.plot(porox,y_pred, color="black")



chfacies=poro.loc[:,'Facies']=="'channel'"
chfacies=poro[chfacies]

poroch=chfacies.loc[:,'Porosity (%)'].to_numpy().reshape(-1,1)
permch=chfacies.loc[:,'Permeability (mD)'].to_numpy().reshape(-1,1)
plotch, ax = plt.subplots()
ax.scatter(poroch,permch)
ax.set_xlabel('Porosity (%)')
ax.set_ylabel('Permeability (mD)')
ax.set_title('Channels')

model = LinearRegression()
model.fit(poroch, permch)
r2ch = model.score(poroch, permch)
ypredch = model.predict(poroch)
ax.plot(poroch,ypredch, color="black")



csfacies=poro.loc[:,'Facies']=="'crevasse splay'"
csfacies=poro[csfacies]

porocs=csfacies.loc[:,'Porosity (%)'].to_numpy().reshape(-1,1)
permcs=csfacies.loc[:,'Permeability (mD)'].to_numpy().reshape(-1,1)
plotcs, ax = plt.subplots()
ax.scatter(porocs,permcs)
ax.set_xlabel('Porosity (%)')
ax.set_ylabel('Permeability (mD)')
ax.set_title('Crevasse Splay')

model = LinearRegression()
model.fit(porocs, permcs)
r2cs = model.score(porocs, permcs)
ypredcs = model.predict(porocs)
ax.plot(porocs,ypredcs, color="black")



obfacies=poro.loc[:,'Facies']=="'overbanks'"
obfacies=poro[obfacies]

obfacies=poro.loc[:,'Facies']=="'overbanks'"
obfacies=poro[obfacies]

poroob=obfacies.loc[:,'Porosity (%)'].to_numpy().reshape(-1,1)
permob=obfacies.loc[:,'Permeability (mD)'].to_numpy().reshape(-1,1)
plotob, ax = plt.subplots()
ax.scatter(poroob,permob)
ax.set_xlabel('Porosity (%)')
ax.set_ylabel('Permeability (mD)')
ax.set_title('Overbanks')

model = LinearRegression()
model.fit(poroob, permob)
r2ob = model.score(poroob, permob)
ypredob = model.predict(poroob)
ax.plot(poroob,ypredob, color="black")



#################### PICTURE ####################

from skimage import io
import matplotlib.pyplot as plt
import numpy as np

img = io.imread('berea8bit.tif')
plt.imshow(img,cmap='gray')
plt.axis('off')
plt.show()
imarray = np.array(img)

imV = imarray.reshape((500*500, 1))
plt.hist(imV, density=True, bins=30, range=[120,255])
plt.title('Porosity Histogram')
plt.show()

BW = imarray
BW[BW<100] = 0
BW[BW>=100] = 255
BW2 = BW.astype(np.bool)
BW2 = np.array(BW2)
plt.imshow(imarray, cmap='Greys_r')
plt.axis('off')

area = np.size(BW2)
fw = np.sum(BW2)

print('Porosity of the thin section:', (1 - (fw/area))*100)
































