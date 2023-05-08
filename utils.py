from matplotlib import cm, colors
import seaborn as sns
# import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager # to solve: Font family ['sans-serif'] not found. Falling back to DejaVu Sans.
import json

centimeter = 1/2.54  # centimeter in inches

# https://www.geeksforgeeks.org/react-js-blueprint-colors-qualitative-color-schemes/
react_cols_10 = ['#147EB3','#29A634','#D1980B','#D33D17','#9D3F9D','#00A396','#DB2C6F','#8EB125','#946638','#7961DB']

# http://tsitsul.in/blog/coloropt/
norm_7 = ['#4053d3','#ddb310','#b51d14','#00beff','#fb49b0','#00b25d','#cacaca']
norm_12 = ['#ebac23','#b80058','#008cf9','#006e00','#00bbad','#d163e6','#b24502',
           '#ff9287','#5954d6','#00c6f8','#878500','#00a76c','#bdbdbd']

def config_rc(dpi=400, font_size=5, lw=1.):
    # matplotlib.rcParams.keys()
    rc={
        'font.size': font_size, 
        'axes.labelsize': font_size, 
        'axes.titlesize': font_size, 
        'xtick.labelsize': font_size, 
        'ytick.labelsize': font_size,
        'figure.dpi':dpi,'axes.linewidth':lw,
    } # 'figure.figsize':(11.7/1.5,8.27/1.5)
    
    sns.set(style='ticks',rc=rc) 
    sns.set_context("paper")

    mpl.rcParams.update(rc)

    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42

    mpl.rcParams['font.sans-serif'] = "Arial"
    mpl.rcParams['font.family'] = "sans-serif"
    mpl.rcParams['axes.unicode_minus']=False # negative minus sign
    

def get_path(key, json_path='./_data.json'):
    with open(json_path, 'r') as f:
        x = json.loads(f.read())
    return x[key]

def _draw_palette(cols):
    sns.color_palette(cols)

