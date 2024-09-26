def import_libraries():
    # Import packages we'll need later on in this tutorial
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import matplotlib.ticker as mtick
    import matplotlib.cm as cm
    import math
    import plotly.graph_objects as go
    from collections import OrderedDict
    from matplotlib.lines import Line2D  # Import for creating custom legend markers
    import json
    import copy
    import random
    import re
    import seaborn as sns
    import importlib

    # Import BW25 packages
    import bw2data as bd
    import bw2io as bi
    import bw2calc as bc
    import bw2analyzer as bwa
    import brightway2 as bw 
    from bw2calc import LeastSquaresLCA

import_libraries()
