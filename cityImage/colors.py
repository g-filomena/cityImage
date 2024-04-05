import numpy as np
import matplotlib.colors as mc
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import colorsys

def random_colors_list(nlabels, vmin = 0.8, vmax = 1.0, hsv = False):
    """ 
    Generates a list of random HSV colors given the number of classes, 
    minimum and maximum values in the HSV spectrum.
    
    Parameters
    ----------
    nlabels: int
        The number of classes or labels for which colors are generated.
    vmin: float
        The minimum value in the HSV spectrum. Default is 0.8.
    vmax: float
        The maximum value in the HSV spectrum. Default is 1.0.
    hsv: bool
        Indicates whether to return colors in HSV format (True) or convert them to RGB format (False).
        Default is False.
    
    Returns
    -------
    colors : list
        A list of random HSV colors or RGB colors, depending on the value of `hsv`.
    """
    randRGBcolors = []
    randHSVcolors = [(np.random.uniform(low=0.0, high=0.95),
                      np.random.uniform(low=0.4, high=0.95),
                      np.random.uniform(low= vmin, high= vmax)) for i in range(nlabels)]
   
    # Convert HSV list to RGB
    if hsv:
        randRGBcolors = []
        for HSVcolor in randHSVcolors: 
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))
    return randHSVcolors  
            
# Generate random colormap
def rand_cmap(nlabels, type_color ='soft'):
    """ 
    It generates a categorical random color map, given the number of classes
    
    Parameters
    ----------
    nlabels: int
        The number of categories to be coloured.
    type_color: str {"soft", "bright"} 
        It defines whether using bright or soft pastel colors, by limiting the RGB spectrum.
       
    Returns
    -------
    cmap: matplotlib.colors.LinearSegmentedColormap
        The color map.
    """   
    if type_color not in ('bright', 'soft'):
        type_color = 'bright'
    
    # Generate color map for bright colors, based on hsv
    if type_color == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=0.8),
                          np.random.uniform(low=0.2, high=0.8),
                          np.random.uniform(low=0.9, high=1.0)) for i in range(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))


        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type_color == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in range(nlabels)]

        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    return random_colormap

def kindlmann():
    """ 
    It returns a Kindlmann color map. See https://ieeexplore.ieee.org/document/1183788
       
    Returns
    -------
    cmap: matplotlib.colors.LinearSegmentedColormap
        The color map.
    """   

    kindlmann_list = [(0.00, 0.00, 0.00,1), (0.248, 0.0271, 0.569, 1), (0.0311, 0.258, 0.646,1),
            (0.019, 0.415, 0.415,1), (0.025, 0.538, 0.269,1), (0.0315, 0.658, 0.103,1),
            (0.331, 0.761, 0.036,1),(0.768, 0.809, 0.039,1), (0.989, 0.862, 0.772,1),
            (1.0, 1.0, 1.0)]
    cmap = LinearSegmentedColormap.from_list('kindlmann', kindlmann_list)
    return cmap
    
def normalize(n, range1, range2):
    """ 
    Normalizes a value `n` from one range to another range.
    
    Parameters
    ----------
    n: float or int
        The value to be normalized.
    range1: tuple or list
        The original range of values [min, max] from which `n` is taken.
    range2: tuple or list
        The target range of values [min, max] to which `n` will be normalized.
    
    Returns
    -------
    float or int
        The normalized value of `n` in the target range.
    """ 
    delta1 = range1[1] - range1[0]
    delta2 = range2[1] - range2[0]
    return (delta2 * (n - range1[0]) / delta1) + range2[0]  
         
def lighten_color(color, amount=0.5):
    """ 
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    
    Parameters
    ----------
    color: str or tuple
        The color to be lightened. It can be a matplotlib color string, hex string, or RGB tuple.
    amount: float, optional
        The amount by which to lighten the color. Default value is 0.5.
    
    Returns
    -------
    tuple
        The lightened color in RGB format.
    """ 
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])