import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc
import matplotlib.gridspec as gridspec
from collections import Counter
import pandas as pd


def plt_style(c="k"):
    """
    Set plotting style for bright (``c = 'w'``) or dark (``c = 'k'``) backgrounds
    :param c: colour, can be set to ``'w'`` or ``'k'`` (which is the default)
    :type c: str
    """

    # Reset previous configuration
    mpl.rcParams.update(mpl.rcParamsDefault)
    # %matplotlib inline  # not from script
    get_ipython().run_line_magic("matplotlib", "inline")

    # configuration for bright background
    if c == "w":
        plt.style.use("bmh")

    # configurations for dark background
    if c == "k":
        # noinspection PyTypeChecker
        plt.style.use(["dark_background", "bmh"])

    # remove background colour, set figure size
    rc("figure", figsize=(16, 8), max_open_warning=False)
    rc("axes", facecolor="none")


def exif_analysis(photos):

    plt.style.use("bmh")
    csfont = {"family": "Arno Pro", "size": 20}
    mpl.rc("font", **csfont)

    fig = plt.figure(figsize=(15, 5))

    ## Exif local length ##
    gs = gridspec.GridSpec(1, 1, figure=fig, hspace=0.4)
    plt.subplot(gs[0, 0])
    pd.value_counts(photos["exif_focal_length"])[:40].plot.bar()
    plt.xticks(fontsize=13)
    plt.grid(None)
    plt.title("Focal length values")
    plt.savefig("figures/focal_values.png")

    ## Aperture ##
    gs = gridspec.GridSpec(1, 1, figure=fig, hspace=0.4)
    plt.subplot(gs[0, 0])
    pd.value_counts(photos["exif_aperture_value"])[:10].plot.bar()
    plt.xticks(fontsize=13)
    plt.grid(None)
    plt.title("Aperture values")
    plt.savefig("figures/aperture_values.png")

    ## Exposure
    gs = gridspec.GridSpec(1, 1, figure=fig, hspace=0.4)
    plt.subplot(gs[0, 0])
    pd.value_counts(photos["exif_exposure_time"])[:40].plot.bar()
    plt.xticks(fontsize=13)
    plt.grid(None)
    plt.title("Exposure values")
    plt.savefig("figures/exposure_values.png")

    ## ISO ##
    gs = gridspec.GridSpec(1, 1, figure=fig, hspace=0.4)
    plt.subplot(gs[0, 0])
    pd.value_counts(photos["exif_iso"])[:40].plot.bar()
    plt.xticks(fontsize=13)
    plt.title("ISO values")
    plt.grid(None)

    plt.savefig("figures/iso_values.png")


def label_distribution(photos, filename):

    plt.style.use("bmh")
    csfont = {"family": "Arno Pro", "size": 20}
    mpl.rc("font", **csfont)
    fig = plt.figure(figsize=(12, 15))

    ## ISO ##
    gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.4)

    plt.subplot(gs[0, 0])

    plt.title("Distribution in Iso Noise Labels - (3bins)", **csfont)
    plt.bar(
        dict(Counter(photos["iso_noise_label"])).keys(),
        dict(Counter(photos["iso_noise_label"])).values(),
    )
    plt.xticks([0, 1, 2])
    plt.grid(None)

    plt.subplot(gs[1, 0])

    plt.title("Distribution in Iso Noise Labels - (2bins)", **csfont)
    plt.bar(
        dict(Counter(photos["iso_noise_bin_label"])).keys(),
        dict(Counter(photos["iso_noise_bin_label"])).values(),
    )
    plt.xticks([0, 1])
    plt.grid(None)
    plt.savefig("figures/iso_bins.png")

    ## DoF ##
    gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.4)

    plt.subplot(gs[0, 0])

    plt.title("Distribution in DoF Labels - (3bins)", **csfont)
    plt.bar(dict(Counter(photos["DoF"])).keys(), dict(Counter(photos["DoF"])).values())
    plt.xticks([0, 1, 2])
    plt.grid(None)

    plt.subplot(gs[1, 0])

    plt.title("Distribution in DoF Labels - (2bins)", **csfont)
    plt.bar(
        dict(Counter(photos["DoF_bin"])).keys(),
        dict(Counter(photos["DoF_bin"])).values(),
    )
    plt.xticks([0, 1])
    plt.grid(None)
    plt.savefig("figures/dof_bins.png")

    ## Exposure ##
    gs = gridspec.GridSpec(1, 1, figure=plt.figure(figsize=(12, 8)), hspace=0.4)

    plt.subplot(gs[0, 0])

    plt.title("Distribution in Exposure Labels - (3bins)", **csfont)
    plt.bar(
        dict(Counter(photos["exposure_label"])).keys(),
        dict(Counter(photos["exposure_label"])).values(),
    )
    plt.xticks([0, 1, 2])
    plt.grid(None)

    plt.savefig("figures/exposure_bins.png")

    ## Focal Length ##
    gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.4)

    plt.subplot(gs[0, 0])

    plt.title("Distribution in Focal Length Labels - (3bins)", **csfont)
    plt.bar(
        dict(Counter(photos["focal_label"])).keys(),
        dict(Counter(photos["focal_label"])).values(),
    )
    plt.xticks([0, 1, 2])
    plt.grid(None)

    plt.subplot(gs[1, 0])

    plt.title("Distribution in Focal Length Labels - (2bins)", **csfont)
    plt.bar(
        dict(Counter(photos["focal_label_bin"])).keys(),
        dict(Counter(photos["focal_label_bin"])).values(),
    )
    plt.xticks([0, 1])

    plt.grid(None)

    plt.savefig("figures/focal_length_bins.png")
