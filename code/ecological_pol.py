import sys
sys.path.append(r'/hr-fs02/hr_projekte/Pol-InSAR_InfoRetrieval/10_users/mans_is/PyPolSAR')


import rioxarray
import rasterio as rio
import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import box, mapping
from scipy import ndimage
from pypolsar import geo
import pprint



import h5py

from pypolsar import polsar
import matplotlib.pyplot as plt

from pypolsar import utils, plot
from pypolsar import polsar
from pypolsar.polsar import decomposition

from pypolsar.stats.timer import Timer
from pathlib import Path
import pypolsar

import seaborn as sns
import matplotlib
from matplotlib.colors import LogNorm
from matplotlib.ticker import PercentFormatter
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap


fig_save = Path("./../figures/")

from pathlib import Path

process_path = Path('./../data/processed/')
# p = Path('./')
prmasr_nc = {}
prmasr_nc["L"] = {}
prmasr_nc["S"] = {}
prmasr_nc["L"]["18"] = sorted(list( process_path.glob('./ecological_masks/18*L.nc')))
prmasr_nc["L"]["19"] = sorted(list( process_path.glob('./ecological_masks/19*L.nc')))
prmasr_nc["S"]["18"] = sorted(list( process_path.glob('./ecological_masks/18*S.nc')))
prmasr_nc["S"]["19"] = sorted(list( process_path.glob('./ecological_masks/19*S.nc')))
prmasr_nc


colors_hex = ["#4DB65B", "#425DAA", "#F1EB58", "#A4AA56", "#49B0AD", "#A95052", "#74C044"]
color_names = ['Avadlek', 'Guillemot', 'Herschel', 'Komakuk', 'Orca', 'Plove+Jae', 'Thrasher']

ecological_colors = {}
for name, color_hex in zip(color_names, colors_hex):
    ecological_colors[name] = matplotlib.colors.to_rgb(color_hex)
    
print(ecological_colors)


def combine_netcdf(path_18, path_19, clip=None):
    ds_1 = xr.open_dataset(path_18)
    ds_2 = xr.open_dataset(path_19)

    ds_con = xr.concat([ds_1, ds_2], dim=pd.Index([18, 19], name='time'))
    ds_con = ds_con.where(ds_con.mask_valid == 1) 
    if clip is not None:
        ds_con = clip_raster_with_shp(ds_con, shapefile= shapefile_clip)
    
    return ds_con

ds_l = combine_netcdf(prmasr_nc["L"]["18"][0], prmasr_nc["L"]["19"][0])
ds_s = combine_netcdf(prmasr_nc["S"]["18"][0], prmasr_nc["S"]["19"][0])
ds_l["offnadir"] = np.rad2deg(ds_l["offnadir"])
ds_s["offnadir"] = np.rad2deg(ds_s["offnadir"])
ds_l["aoi"] = np.rad2deg(ds_l["aoi"])
ds_s["aoi"] = np.rad2deg(ds_s["aoi"])

# convert to dataframe
ds_l_df = ds_l.to_dataframe()
ds_s_df = ds_s.to_dataframe()


unit_masks = ['Avadlek', 'Guillemot', 'Herschel', 'Komakuk', 'Orca', 'Plove_+Jae', 'Thrasher']
unit_names = ['Avadlek', 'Guillemot', 'Herschel', 'Komakuk', 'Orca', 'Plove+Jae', 'Thrasher']

for UnitName, UnitMask in zip(unit_names, unit_masks):
    print(UnitName, UnitMask)
    ds_l_df.loc[ds_l_df['mask_' + UnitMask ]==1,'UnitName'] = UnitName
    ds_s_df.loc[ds_s_df['mask_' + UnitMask ]==1,'UnitName'] = UnitName


overlap_classes_l = ds_l_df['mask_' + unit_masks[0] ] + \
    ds_l_df['mask_' + unit_masks[1] ] + \
    ds_l_df['mask_' + unit_masks[2] ] + \
    ds_l_df['mask_' + unit_masks[3] ] + \
    ds_l_df['mask_' + unit_masks[4] ] + \
    ds_l_df['mask_' + unit_masks[5] ] > 1
ds_l_df.loc[overlap_classes_l==1,'UnitName'] = np.NaN 

overlap_classes_s = ds_s_df['mask_' + unit_masks[0] ] + \
    ds_s_df['mask_' + unit_masks[1] ] + \
    ds_s_df['mask_' + unit_masks[2] ] + \
    ds_s_df['mask_' + unit_masks[3] ] + \
    ds_s_df['mask_' + unit_masks[4] ] + \
    ds_s_df['mask_' + unit_masks[5] ] > 1

ds_s_df.loc[overlap_classes_s==1,'UnitName'] = np.NaN 

np.count_nonzero(overlap_classes_l), np.count_nonzero(overlap_classes_s)


def plot_2dhist_sns(df, col='UnitName', row="time", x_hist='aoi', 
                    y_hist='entropy', x_y_label_dict=None, 
                    xlim=None, ylim=None):
    sns.set(font_scale=1.85)  # crazy big
    sns.set_style("white")

    df = df.reset_index()
    df = df.loc[~pd.isnull(df.UnitName)]
    g = sns.FacetGrid(df.reset_index(), col=col, row=row, height=6, aspect=1, 
                      hue="UnitName", palette=ecological_colors, 
                      col_order=unit_names, hue_order=unit_names, 
                      xlim=xlim, ylim=ylim)
    g.map(sns.histplot, x_hist, y_hist, bins=1000, )
    if x_y_label_dict is not None :
        g.set_axis_labels(x_y_label_dict[x_hist], x_y_label_dict[y_hist])

    return g



latex_label = {'entropy': 'Entropy (${H}_w$)', 
               'offnadir': 'offnadir [deg]',
               'anisotropy': 'Anisotropy (${H}_w$)', 
               'alpha': 'Alpha (${\\alpha}_{avg}$) [deg]',
               'p_hhvv': '${P}_{HHVV}$',
               'ph_diff_hhvv': 'CPD ${Phase}_{HHVV}$ [deg]',
               'aoi': 'AOI [deg]',}
latex_label["entropy"]


pol_key = ['entropy', 'anisotropy', 'alpha', 'p_hhvv', 'ph_diff_hhvv']
pol_key

x_hist='entropy'


for item in pol_key:
    print(item)
    if x_hist != item:
        file_name = prmasr_nc["L"]["18"][0].name[:-3] + "_" + prmasr_nc["L"]["19"][0].name[:-3]
        pol_key_name = item
        if pol_key_name == "p_hhvv":
            ylim=(0.5, 1.5)
        elif pol_key_name == "ph_diff_hhvv":
            ylim=(-20, 40)
        else:
            ylim=None
        fig = plot_2dhist_sns(df=ds_l_df, 
                         x_hist=x_hist, y_hist=pol_key_name, ylim=ylim,
                        col='UnitName', row="time", x_y_label_dict=latex_label)

        fig.savefig(fig_save.joinpath('PNG', "2d_hist", file_name + "_" + 
                                      pol_key_name + '-' + x_hist+ ".png"))
        plt.clf()
    
    

for item in pol_key:
    print(item)
    if x_hist != item:
        file_name = prmasr_nc["S"]["18"][0].name[:-3] + "_" + prmasr_nc["S"]["19"][0].name[:-3]
        pol_key_name = item
        if pol_key_name == "p_hhvv":
            ylim=(0.5, 1.5)
        elif pol_key_name == "ph_diff_hhvv":
            ylim=(-20, 40)
        else:
            ylim=None
        fig = plot_2dhist_sns(df=ds_l_df, 
                         x_hist=x_hist, y_hist=pol_key_name, ylim=ylim,
                        col='UnitName', row="time", x_y_label_dict=latex_label)

        fig.savefig(fig_save.joinpath('PNG', "2d_hist", file_name + "_" + 
                                      pol_key_name + '-' + x_hist+ ".png"))
        plt.clf()


