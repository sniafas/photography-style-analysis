import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc
import matplotlib.gridspec as gridspec
from collections import Counter
import pandas as pd

def plt_style(c='k'):
    """
    Set plotting style for bright (``c = 'w'``) or dark (``c = 'k'``) backgrounds
    :param c: colour, can be set to ``'w'`` or ``'k'`` (which is the default)
    :type c: str
    """

    # Reset previous configuration
    mpl.rcParams.update(mpl.rcParamsDefault)
    # %matplotlib inline  # not from script
    get_ipython().run_line_magic('matplotlib', 'inline')

    # configuration for bright background
    if c == 'w':
        plt.style.use('bmh')

    # configurations for dark background
    if c == 'k':
        # noinspection PyTypeChecker
        plt.style.use(['dark_background', 'bmh'])

    # remove background colour, set figure size
    rc('figure', figsize=(16, 8), max_open_warning=False)
    rc('axes', facecolor='none')

def exif_analysis(photos, filename):
    plt_style('w')
    fig = plt.figure(figsize=(20, 25))

    gs = gridspec.GridSpec(4, 1, figure=fig, hspace=.4)

    plt.subplot(gs[0, 0])
    pd.value_counts(photos['exif_focal_length'])[:40].plot.bar()
    plt.title('Focal length values', fontsize=12)

    plt.subplot(gs[1, 0])
    pd.value_counts(photos['exif_aperture_value'])[:10].plot.bar()
    plt.title('Aperture values', fontsize=12)

    plt.subplot(gs[2, 0])
    pd.value_counts(photos['exif_exposure_time'])[:40].plot.bar()
    plt.title('Exposure values', fontsize=12)

    plt.subplot(gs[3, 0])
    pd.value_counts(photos['exif_iso'])[:40].plot.bar()
    plt.title('ISO values', fontsize=12)
    #plt.savefig(filename+'.png')

def label_distribution(photos, filename):
    plt_style('w')
    fig = plt.figure(figsize=(12, 15))

    gs = gridspec.GridSpec(6, 1, figure=fig, hspace=.4)

    plt.subplot(gs[0, 0])

    plt.title("Distribution in Iso Noise Labels")
    plt.bar(dict(Counter(photos['iso_noise_label'])).keys(), dict(
        Counter(photos['iso_noise_label'])).values())

    plt.subplot(gs[1, 0])

    plt.title("Distribution in Binary Iso Noise Labels")
    plt.bar(dict(Counter(photos['iso_noise_bin_label'])).keys(), dict(
        Counter(photos['iso_noise_bin_label'])).values())

    plt.subplot(gs[2, 0])

    plt.title("Distribution in DoF Labels")
    plt.bar(dict(Counter(photos['DoF'])).keys(),
            dict(Counter(photos['DoF'])).values())

    plt.subplot(gs[3, 0])

    plt.title("Distribution in Binary DoF Labels")
    plt.bar(dict(Counter(photos['DoF_bin'])).keys(),
            dict(Counter(photos['DoF_bin'])).values())


    plt.subplot(gs[4, 0])

    plt.title("Distribution in Exposure Labels")
    plt.bar(dict(Counter(photos['exposure_label'])).keys(), dict(
        Counter(photos['exposure_label'])).values())

    plt.subplot(gs[5, 0])

    plt.title("Distribution in Focal Length Labels")
    plt.bar(dict(Counter(photos['focal_label'])).keys(), dict(
        Counter(photos['focal_label'])).values())

    #plt.savefig(filename+'.png')
