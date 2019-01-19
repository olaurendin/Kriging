import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from decimal import Decimal
from scipy.stats import norm, probplot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import pandas as pd
from pandas import DataFrame
from geostatsmodels import geoplot, variograms
from pykrige import variogram_models, core


sns.set()

variogram_dict = {'linear': variogram_models.linear_variogram_model,
                  'power': variogram_models.power_variogram_model,
                  'gaussian': variogram_models.gaussian_variogram_model,
                  'spherical': variogram_models.spherical_variogram_model,
                  'exponential': variogram_models.exponential_variogram_model,
                  'hole-effect': variogram_models.hole_effect_variogram_model}

def customreadtext(file, sep, path = False):
    """Extraction function from a textfile

    A lot of entries of pandas.read_csv are not used in this function. If the
    present function does not extract the data properly, please find the
    pandas.read_csv documentation here :
    pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv
    and modify the function appropriately.
    Tested with a textfile with a .txt file with multiple spaces as separators
    and a one-line header.

    Parameters
    ----------
    file : str
        Name of the textfile to open with its extension (".txt")
    sep : int
        Separator used in the text file to distinguish data from one another
    path : str, default : False
        If not False, gives the path where to look for the text file

    Returns
    -------
    P : ndarray,
        packed data in 3 columns :  (x,y,depth_seabed)
    """

    if path:
        # Add the text file path to the name of the textfile in order to find it
        path = os.path.join(path, file)
    else:
        path = file     # Look for the file in the same folder than where this python program is
    if file[:7] != "notide_":                   # If the data is the original one, no preprocessing applied
        z = DataFrame(pd.read_csv(path, header=0,delim_whitespace=True, skipinitialspace=True,names=["t", "phi", "dphi", "theta", "dtheta", "psi", "dpsi", "vx", "dvx", "vy", "dvy", "vz", "dvz", "depth", "ddepth", "a", "da", "x", "dx", "y", "dy"]))
        z['depth_seabed'] = z['depth']+z['a']   # depth of the seabed = depth of the AUV + distance between AUV and seabed
    else:       # in this case the depth of the seabed was calculated during the preprocessing of the data
        z = DataFrame(pd.read_csv(path, header=0,delim_whitespace=True, skipinitialspace=True,names=["t", "phi", "dphi", "theta", "dtheta", "psi", "dpsi", "vx", "dvx", "vy", "dvy", "vz", "dvz", "depth", "ddepth", "a", "da", "x", "dx", "y", "dy","depth_seabed"]))
    z = z.iloc[[i for i in range(0,z.shape[0],sep)],:]
    P = np.array( z[['x','y','depth_seabed']] ) # Pack all the data in a single numpy ndarray
    return P

def save_np_array(txtfile, data, path="."):
    """Save a numpy array as a text file, erases any text file with the same name previously
    existing in the selected directory.

    Parameters
    ----------
    txtfile : str
        Name of the textfile to write into with its extension (".txt")
    data : numpy array
        Data to save in the text file.
    path : str, default : "."
        Gives the path where to look for the text file. Points to the current
        directory by default.
    """
    if not isinstance(txtfile, str):
        print("Please enter a text file name as a string")
        os.abort()
    file = os.path.join(path, txtfile)
    if txtfile in os.listdir(path):
        open(file, 'w').close()
    np.savetxt(file, data)

def save_fig(namefile, suffix):
    if namefile[-4]==".":
        plt.savefig(namefile[:-4]+suffix)
    else:
        plt.savefig(namefile+suffix)

def plotscatterdata(data, namefile = False, xlabel=False, ylabel=False, zlabel=False, cblabel=False, \
                axisequal = False, title = False, wsizeinches = False, cmap = False, \
                fig=False, ax=False):
    """Plots the scatterplot of a 2D or 3D dimensional array"""
    if data.shape[1] == 3:
        fig, ax = plotscatterdata2D(data, xlabel=xlabel, ylabel=ylabel, cblabel=cblabel, axisequal = axisequal, \
                        title = title, wsizeinches = wsizeinches, cmap = cmap, \
                        namefile = namefile, fig=fig, ax=ax)
    elif data.shape[1] == 4:
        fig, ax = plotscatterdata3D(data, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, cblabel=cblabel,\
                        axisequal = axisequal, title = title, wsizeinches = wsizeinches,\
                        cmap = cmap, namefile = namefile, fig=fig, ax=ax)
    else:
        raise ValueError("matrix data of incompatible shape for plotscatterdata.\
            Data must be of shape (N,M+1) where N stands for the number of points\
            and M the number of dimensions for the training points. So (N,3) for\
            2 dimensional kriging or (N,4) for 3 dimensional kriging.")
    return fig, ax

def plotscatterdata2D(data, xlabel=False, ylabel=False, cblabel=False, axisequal = False, \
                title = False, wsizeinches = False, cmap = False, \
                namefile = False, fig=False, ax=False):
    """Plot the scatterplot of a two dimensional ndarray

    Parameters
    ----------
    data : ndarray, (N,3)
        The first two dimensions are the position of each point,
        The third dimension encodes the colour of each point
    xlabel : str
        The label of the x axis. None if False
    ylabel : str
        The label of the y axis. None if False
    cblabel : str
        The label of the colorbar. None if False
    axisequal : bool
        If True, x and y axis are of same length
    title : str
        Title of the scatterplot, None if False
    wsizeinches : tuple (1,2)
        Size of the window in inches horizontally and vertically
    cmap : matplotlib.colors.LinearSegmentedColormap or RGB tuple or matplotlib colour map
        The list of colors for the third dimensional variable
    namefile : str, default : False
        If not False, gives the path and name of the screenshot taken automatically
    fig : matplotlib.pyplot.figure
        If not False, gives the matplotlib figure instance on which add the plot.
        If given, must be given along ax
    ax : Axes3D
        If not False, gives the matplotlib Axes3D instance on which add the plot.
        If given, must be given along fig

    Returns
    -------
    fig : matplotlib.pyplot.figure
        Returned for possible addition to the plot outside of the function
    ax : Axes3D
        Returned for possible addition to the plot outside of the function
    """

    if not fig :
        fig = plt.figure()
    if not ax :
        ax = Axes3D(fig)
    if not wsizeinches:
        wsizeinches = (8,8)
    if not cmap :
        cmap = geoplot.YPcmap
    fig.set_size_inches(wsizeinches)
    datax, datay, dataz = np.split(data,data.shape[1], axis=1)  # get columns
    s = plt.scatter( datax, datay, c=dataz, s=64,cmap=cmap)
    if axisequal:
        plt.axis('equal')
    cb = plt.colorbar(s)
    if cblabel:
        cb.set_label(cblabel)
    if xlabel :
        plt.xlabel(xlabel)
    if ylabel :
        plt.ylabel(ylabel)
    if title :
        plt.title(title)
    if namefile :
        save_fig(namefile, "_scatter.png")
    return fig, ax

def plotscatterdata3D(data, xlabel=False, ylabel=False, zlabel=False, cblabel=False, \
                axisequal = False, title = False, wsizeinches = False, cmap = False, \
                namefile = False, fig=False, ax=False):
    """Plot the scatterplot of a three dimensional ndarray

    Parameters
    ----------
    data : ndarray, (N,4)
        The first three dimensions are the position of each point,
        The fourth dimension encodes the colour of each point
    xlabel : str
        The label of the x axis. None if False
    ylabel : str
        The label of the y axis. None if False
    zlabel : str
        The label of the z axis. None if False
    cblabel : str
        The label of the colorbar. None if False
    axisequal : bool
        If True, x and y axis are of same length
    title : str
        Title of the scatterplot, None if False
    wsizeinches : tuple (1,2)
        Size of the window in inches horizontally and vertically
    cmap : matplotlib.colors.LinearSegmentedColormap or RGB tuple or matplotlib colour map
        The list of colors for the third dimensional variable
    namefile : str, default : False
        If not False, gives the path and name of the screenshot taken automatically
    fig : matplotlib.pyplot.figure
        If not False, gives the matplotlib figure instance on which add the plot.
        If given, must be given along ax
    ax : Axes3D
        If not False, gives the matplotlib Axes3D instance on which add the plot.
        If given, must be given along fig

    Returns
    -------
    fig : matplotlib.pyplot.figure
        Returned for possible addition to the plot outside of the function
    ax : Axes3D
        Returned for possible addition to the plot outside of the function
    """
    if not fig :
        fig = plt.figure()
    if not ax :
        ax = Axes3D(fig)
    if not wsizeinches:
        wsizeinches = (8,8)
    if not cmap :
        cmap = geoplot.YPcmap
    fig.set_size_inches(wsizeinches)
    datax, datay, dataz, datat = np.split(data,data.shape[1], axis=1)  # get columns
    s = ax.scatter( datax, datay, dataz, c=datat[:,0], s=4,cmap=cmap)
    ax.view_init(elev=90., azim=0.)
    if axisequal:
        plt.axis('equal')
    cb = plt.colorbar(s)
    if cblabel:
        cb.set_label(cblabel)
    if xlabel :
        ax.set_xlabel(xlabel)
        ax.xaxis.labelpad=15
    if ylabel :
        ax.set_ylabel(ylabel)
        ax.yaxis.labelpad=15
    if zlabel :
        ax.set_zlabel(zlabel)
        ax.zaxis.labelpad=15
    if title :
        ax.set_title(title)
    if namefile :
        save_fig(namefile, "_scatter3D.png")
    return fig, ax

def plotgaussiandist(dataz, namefile = False, bins = 6, xlabel=False, fig=False):
    """Plot the comparison between the spatial repartition of the data and a gaussian repartition

    This is composed of two subplots giving the histogram of the density of the
    z component of the data and the corresponding best gaussian fit, and a comparative
    plot between the experimental quartiles of the data and the theoretical
    quartiles of a given gaussian distribution. Both of these plot tend to show
    the closeness of the data to a gaussian distribution. The idea being that the
    closer the data is from a random gaussian distribution, the better the accuracy
    of the kriging algorithm.
    Each of these plots are taken care of by subroutines.

    Parameters
    ----------
    dataz : ndarray, (N,1)
        The data to compare to a gaussian distribution
    bins : int, default : 6
        Number of bins used to create the histogram. The bigger it is the better
        the accuracy of the gaussian fit is.
    namefile : str, default : False
        If not False, gives the path and name of the screenshot taken automatically
    fig : matplotlib.pyplot.figure
        If not False, gives the matplotlib figure instance on which add the plot.
    Returns
    -------
    fig : matplotlib.pyplot.figure
        The matplotlib figure used to plot the gaussianfit. Is returned for extra
        addition of other plots on the same figure.
    mu : float
        the mean of the gaussian fit of dataz
    std : float
        the standard deviation of the gaussian fit of dataz
    """
    if not fig:
        fig = plt.figure()
    plt.subplot(211)
    mu,std = plotgaussiandistfit(dataz, bins = bins, xlabel= xlabel, alpha = 0.6, color="c")
    plt.subplot(212)
    plotgaussiandistprobplot(dataz)
    if namefile :
        save_fig(namefile, "_gaussiandist.png")
    fig.tight_layout()  # so that the x label of the first figure doesn't overlap
                        # the title of the second figure.
    return fig, mu, std

def plotgaussiandistfit(dataz, bins = 6, xlabel=False, alpha = 0.6, color="c"):
    """Plot the  histogram of the density of the dataz and the corresponding best gaussian fit

    Parameters
    ----------
    dataz : ndarray, (N,1)
        The data to compare to a gaussian distribution
    bins : int, default : 6
        Number of bins used to create the histogram. The bigger it is the better
        the accuracy of the gaussian fit is.
    alpha : float, default : 0.6
        Transparency of the histogram. 0 is fully transparent, 1 fully opaque
    color : str, default : "c"
        color of the histogram. Could be a single letter ("c" for cyan, "k" for
        black ...), a string ("yellow") or an RGB tuple (255,255,255). See
        matplotlib documentation for further details :
        https//matplotlib.org/tutorials/index.html#tutorials-color
    Returns
    -------
    mu : float
        the mean of the gaussian fit of dataz
    std : float
        the standard deviation of the gaussian fit of dataz
    """
    mu, std = norm.fit(dataz)
    plt.hist(dataz, bins=bins, normed=True, alpha=alpha, color=color)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    plt.title("Fit results: mu = %.5E,  std = %.5E" % (Decimal(mu), Decimal(std)))
    plt.ylabel('Density')
    if xlabel:
        plt.xlabel(xlabel)
    return mu, std

def plotgaussiandistprobplot(dataz):
    """comparative
    plot between the experimental quartiles of the data and the theoretical quartiles of a given gaussian distribution

    Parameters
    ----------
    dataz : ndarray, (N,1)
        The data to compare to a gaussian distribution
    """
    qqdata = probplot(dataz, dist="norm",plot=plt,fit=False)
    plt.gcf()

def plotresidualsvario(delta, sigma, epsilon, namefile = False):
    """Plot to visualize the accuracy of the variogram found through cross validation.

    Plot the histograms of the residuals, mean squared error and standardized residuals
    between an estimation at the training data points and the original data.
    Also plots the histograms and best gaussian fit of each.

    Parameters
    ----------
    delta : ndarray, (N,1)
        The residuals between the estimated data obtained through cross validation
        and the original data plotted with regard to the indexes of the points.
    sigma : ndarray, (N,1)
        The mean squared error between the estimated data obtained through cross validation
        and the original data plotted with regard to the indexes of the points.
    epsilon : ndarray, (N,1)
        The standardized error between the estimated data obtained through cross validation
        and the original data plotted with regard to the indexes of the points.
    namefile : str, default : False
        If not False, gives the path and name of the screenshot taken automatically
    """
    L = np.arange(0,delta.shape[0])
    fig = plt.figure()
    fig.set_size_inches(16,16)
    plt.subplot(321)
    plt.bar(L,delta,width=1)
    plt.ylabel('Residuals')
    plt.subplot(323)
    plt.bar(L,sigma,width=1)
    plt.ylabel('Mean Squared Error')
    plt.subplot(325)
    plt.bar(L,epsilon,width=1)
    plt.ylabel('Residuals/MSE')
    plt.subplot(322)
    plotgaussiandistfit(delta, bins = 15)
    plt.subplot(324)
    plotgaussiandistfit(sigma, bins = 15)
    plt.subplot(326)
    plotgaussiandistfit(epsilon, bins = 15)
    if namefile :
        save_fig(namefile, "_residualsvario.png")
    fig.tight_layout()
    plt.show()

def plot3Dplane(gridx,gridy,est, std=False,P=False, namefile = False, fig=False, ax=False):
    """Plot the 3D surface of the estimations at each points of a regular grid

    Plot the 3D surface of the expectation of the estimated data on a regular
    grid obtained by kriging. Could also plot the standard deviation at each point
    and the original data points if given.

    Parameters
    ----------
    P : ndarray, (N,3)
        The first two dimensions are the position of each training point,
        The third dimension encodes the height of each training point
    gridx : ndarray, (1,N)
        Indices on the x axis used to create the regular grid.
    gridy : ndarray, (1,N)
        Indices on the y axis used to create the regular grid.
    est : ndarray, (N,N)
        Expectation at each given point of the grid obtained through kriging
    std : ndarray, (N,N)
        Standard deviation at each given point of the grid obtained through kriging
    namefile : str, default : False
        If not False, gives the path and name of the screenshot taken automatically
    fig : matplotlib.pyplot.figure
        If not False, gives the matplotlib figure instance on which add the plot.
        If given, must be given along ax
    ax : Axes3D
        If not False, gives the matplotlib Axes3D instance on which add the plot.
        If given, must be given along fig

    Returns
    -------
    fig : matplotlib.pyplot.figure
        Returned for possible addition to the plot outside of the function
    ax : Axes3D
        Returned for possible addition to the plot outside of the function
    """
    if not fig:
        fig = plt.figure()
    if not ax :
        ax = Axes3D(fig)
    xx,yy = np.meshgrid(gridx,gridy)
    if P.__class__.__name__ in ["ndarray", "MaskedArray"]:       # Plot the training points
        ax.scatter(P[:,0], P[:,1], P[:,2], "red", alpha = 1, marker="^")
    ax.plot_surface(xx, yy,est, color ="b", alpha = 0.5, cmap = "jet")  # Plot the estimate surface
    ax.plot_wireframe(xx, yy, est, rstride=5, cstride=5, color="grey")
    if std.__class__.__name__ in ["ndarray", "MaskedArray"]:    # Plot the est+std  ans est-std surfaces
        ax.plot_surface(xx,yy,(est+std), color="r", alpha = 0.2)
        ax.plot_surface(xx,yy,(est-std), color="r", alpha = 0.2)
    if namefile :
        save_fig(namefile, "_map3D.png")
    return fig, ax

def plotcolormesh_estvar(P,gridx,gridy,est, var, nbins = 15, typeplot = "pcolormesh", namefile = False,
                fig=False, ax0=False, ax1=False):
    """Plot the colormesh of the estimations and the standard deviation at each points of a regular grid

    Plot the colormesh of the estimations and the standard deviation at each points
    of a regular grid obtained by kriging. The expectation is represented from
    purple to green according to a linear colorbar. The expectation at each training
    point is also given by the color inside each marker "o". The standard is
    represented from red to blue according to a linear colorbar. The training points
    are represented as black points.

    Parameters
    ----------
    P : ndarray, (N,3)
        The first two dimensions are the position of each point,
        The third dimension encodes the colour of each point
    gridx : ndarray, (1,N)
        Indices on the x axis used to create the regular grid.
    gridy : ndarray, (1,N)
        Indices on the y axis used to create the regular grid.
    est : ndarray, (N,N)
        Expectation at each given point of the grid obtained through kriging
    var : ndarray, (N,N)
        Variance at each given point of the grid obtained through kriging
    nbins : int, default : 15
        Number of levels to consider on the colormap. The bigger the more precise
        the colormap is, the more levels will be visible on the colormesh
    typeplot : str
        Two values possible :
        "pcolormesh" : pixellated colormesh
        "contourf" : continous colormesh
        One value has to be taken for the both the expectation and the standard
        deviation plots.
    namefile : str, default : False
        If not False, gives the path and name of the screenshot taken automatically
    fig : matplotlib.pyplot.figure
        If not False, gives the matplotlib figure instance on which add the plot.
        If given, must be given along ax0 and ax1
    ax0 : Axes3D
        If not False, gives the matplotlib Axes3D instance on which add the first plot.
        If given, must be given along fig and ax1
    ax1 : Axes3D
        If not False, gives the matplotlib Axes3D instance on which add the second plot.
        If given, must be given along fig and ax0

    Returns
    -------
    fig : matplotlib.pyplot.figure
        Returned for possible addition to the plot outside of the function
    ax : Axes3D
        Returned for possible addition to the plot outside of the function
    """
    if not fig and not ax0 and not ax1:
        fig, (ax0, ax1) = plt.subplots(nrows=2)
    xx,yy = np.meshgrid(gridx,gridy)
    # pick the desired colormap
    cmapest = plt.get_cmap('PiYG')
    cmapvar = plt.get_cmap('RdBu')

    plotcolormesh(fig, ax0, xx, yy, est, cmap=cmapest, nbins=nbins, typeplot=typeplot)
    ax0.scatter(P[:,0],P[:,1],s=16, edgecolor='k', c = P[:,2], cmap = cmapest)
    ax0.set_title(typeplot+' of expectation')

    plotcolormesh(fig, ax1, xx, yy, var, cmap=cmapvar, nbins=nbins, typeplot=typeplot)
    ax1.scatter(P[:,0],P[:,1],s=4, c = "k")
    ax1.set_title(typeplot+' of mean squared error')

    # adjust spacing between subplots so `ax1` title and `ax0` tick labels
    # don't overlap
    fig.tight_layout()
    if namefile :
        save_fig(namefile, "_colormesh.png")
    return fig, ax0, ax1

def plotcolormesh_estvar_fakedeposits(P,deposits,gridx,gridy,est, var, nbins = 15,
        typeplot = "pcolormesh", namefile = False, fig=False, ax0=False, ax1=False):
    """Plot the colormesh of the estimations and the standard deviation at each points of a regular grid

    Plot the colormesh of the estimations and the standard deviation at each points
    of a regular grid obtained by kriging. The expectation is represented from
    purple to green according a linear colorbar. The expectation at each training
    point is no longer given by the color inside each marker "o". In fact, these are
    replaced by simple "." markers. However the position of the fake deposits
    are shown in red. The standard is represented from red to blue according a
    linear colorbar. The training points are represented as black points.

    Parameters
    ----------
    P : ndarray, (N,3)
        The first two dimensions are the position of each point,
        The third dimension encodes the colour of each point
    gridx : ndarray, (1,N)
        Indices on the x axis used to create the regular grid.
    gridy : ndarray, (1,N)
        Indices on the y axis used to create the regular grid.
    est : ndarray, (N,N)
        Expectation at each given point of the grid obtained through kriging
    var : ndarray, (N,N)
        Standard deviation at each given point of the grid obtained through kriging
    nbins : int, default : 15
        Number of levels to consider on the colormap. The bigger the more precise
        the colormap is, the more levels will be visible on the colormesh
    typeplot : str
        Two values possible :
        "pcolormesh" : pixellated colormesh
        "contourf" : continous colormesh
        One value has to be taken for the both the expectation and the standard
        deviation plots.
    namefile : str, default : False
        If not False, gives the path and name of the screenshot taken automatically
    fig : matplotlib.pyplot.figure
        If not False, gives the matplotlib figure instance on which add the plot.
        If given, must be given along ax0 and ax1
    ax0 : Axes3D
        If not False, gives the matplotlib Axes3D instance on which add the first plot.
        If given, must be given along fig and ax1
    ax1 : Axes3D
        If not False, gives the matplotlib Axes3D instance on which add the second plot.
        If given, must be given along fig and ax0

    Returns
    -------
    fig : matplotlib.pyplot.figure
        Returned for possible addition to the plot outside of the function
    ax : Axes3D
        Returned for possible addition to the plot outside of the function
        """
    if not fig and not ax0 and not ax1:
        fig, (ax0, ax1) = plt.subplots(nrows=2)
    xx,yy = np.meshgrid(gridx,gridy)

    # pick the desired colormap
    cmapest = plt.get_cmap('PiYG')
    cmapvar = plt.get_cmap('RdBu')

    plotcolormesh(fig, ax0, xx, yy, est, cmap=cmapest, nbins=nbins, typeplot=typeplot)
    ax0.scatter(P[:,0],P[:,1],s=4, c = "k")
    ax0.scatter(deposits[:,0], deposits[:,1], c="r", s=16*deposits[:,3])
    ax0.set_title(typeplot+' of expectation')

    plotcolormesh(fig, ax1, xx, yy, var, cmap=cmapvar, nbins=nbins, typeplot=typeplot)
    ax1.scatter(P[:,0],P[:,1],s=4, c = "k")
    ax1.set_title(typeplot+' of mean squared error')

    # adjust spacing between subplots so `ax1` title and `ax0` tick labels
    # don't overlap
    fig.tight_layout()
    if namefile :
        save_fig(namefile, "_colormesh.png")
    return fig, ax0, ax1

def plotcolormesh(fig, ax, xx, yy, z, cmap, nbins=15, typeplot="pcolormesh"):
    """Plot the colormesh of a given z data with regard to the meshgrid (xx,yy).

    Parameters
    ----------
    fig : matplotlib.pyplot.figure instance
        Instance of a plt.figure() on which to apply the colormesh
    ax : matplotlib.axes.Axes instance
        Instance of matplotlib.axes.Axes instance on which to apply the colormesh
    xx : ndarray, (N,N)
        x meshgrid, a.k.a the x indexes of the regular grid
    yy : ndarray, (N,N)
        y meshgrid, a.k.a the y indexes of the regular grid
    z : ndarray, (N,1)
        data encoding the color of the colormesh
    cmap : LinearSegmentedColormap
        Colormap defining the colors of the meshgrid*
    nbins : int, default : 15
        Number of levels to consider on the colormap. The bigger the more precise
        the colormap is, the more levels will be visible on the colormesh
    typeplot : str
        Two values possible :
        "pcolormesh" : pixellated colormesh
        "contourf" : continous colormesh
        One value has to be taken for the both the expectation and the standard
        deviation plots.
    """

    # pick the sensible levels and define a normalization
    # instance which takes data values and translates those into levels.
    levels = MaxNLocator(nbins=nbins).tick_values(z.min(), z.max())
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    if typeplot == "pcolormesh":    # Pixellated colormesh
        im = ax.pcolormesh(xx, yy, z, cmap=cmap, norm=norm)
    elif typeplot == "contourf":      # Continuous colormesh
        im = ax.contourf(xx,yy, z, levels=levels,cmap=cmap)
    else:
        print("Type {} of colormesh not recognized".format(typeplot))
    fig.colorbar(im, ax=ax)

def polaranisotropy(data, pwdist, lags, tol, nsectors, angldev):
    """
    data : ndarray, (N,3)
        The first two dimensions are the position of each point,
        The third dimension encodes the value of each point
    pwdist : numpy array
        Pairwise distances of the data points
    lags : list or numpy array
        Centers of the lags
    tol : float or integer
        Length tolerance of the sectors
    nsectors : integer
        Number of directions of the sectors.
    angldev : float or integer
        Angle deviation from North from which all sectors are created
    """
    angle = 180.0 / nsectors
    atol = angle / 2.0
    sectors = [atol +angldev+ i * angle for i in range(nsectors)]
    varios = []
    lenlags = []
    for i,sector in enumerate(sectors):
        varios.append([])
        lenlags.append([])
        for lag in lags:
            anisodata = (data, pwdist, lag, tol, sector, atol)
            indices = variograms.anilagindices(*anisodata)
            sv = variograms.semivariance(data, indices)
            varios[i].append(sv)
            lenlags[i].append(len(indices))
    return varios, lenlags, sectors

def printparamsvario(paramsvario, modelvario, postxt,fontsize):
    """
    Prints the variogram parameters given the type of variogram
    """
    print("Using '%s' Variogram Model" % str(modelvario))
    if modelvario == 'linear':
        print("Slope:", paramsvario[0])
        print("Nugget:", paramsvario[1], '\n')
        plt.text(0,2,"Slope : {}\nNugget : {}".format\
                    (paramsvario[0], paramsvario[1]))
    elif modelvario == 'power':
        print("Scale:", paramsvario[0])
        print("Exponent:", paramsvario[1])
        print("Nugget:", paramsvario[2], '\n')
        plt.text(0,2,"Scale : {}\n Exponent : {}\nNugget : {}".format\
                    (paramsvario[0], paramsvario[1], paramsvario[1]))
    else:
        print("Partial Sill:", paramsvario[0])
        print("Full Sill:", paramsvario[0] + paramsvario[2])
        print("Range:", paramsvario[1])
        print("Nugget:", paramsvario[2], '\n')
        plt.text(postxt[0], postxt[1],\
                "Partial Sill : {:.2f}\nRange : {:.0f}\nNugget : {:.0f}".format\
                (paramsvario[0], paramsvario[1], paramsvario[2]), fontsize = fontsize)

def plotanglevarios(P, pw, lags, tol, nsector, xlim, prec = 0.1, \
                    angldev=0, degpoly=0, variofit = False, namefile = False):
    """
    Plots the variograms in all directions to detect the presence of an angle
    anisotropy in the data.
    """
    varios, lenlags, sectors = polaranisotropy(P, pw, lags, tol=tol, nsectors=nsector, angldev=angldev)
    d = np.arange(len(varios[0]))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    fig, ax = plt.subplots()
    plt.subplot(211)
    for i,line in enumerate(varios):
        angl = (i+1) * 180./ nsector +angldev
        atole = 90./nsector
        vi = varios[i]
        icol = i%len(colors)
        plt.plot(lags, vi, colors[icol], label="angle = {} +/- {}".format(angl, atole))
        vi=np.array(vi)[~np.isnan(vi)]
        lenvi=vi.shape[0]
        if degpoly>0:
            params = np.polyfit(lags[:lenvi], vi,degpoly)
            fit = np.polyval(params, lags)
            plt.plot(lags, fit, colors[icol]+":", label="fit deg {}".format(params.shape[0]-1))
            plt.plot(lags[:lenvi], vi-fit[:lenvi], colors[icol]+"--", label="result")
        if variofit:
            variomodel = "spherical"
            variofunc = variogram_dict[variomodel]
            variomodelparams = core._calculate_variogram_model(lags[:lenvi], varios[i][:lenvi], \
            variomodel,variofunc, weight=True)
            lags2 = np.arange(xlim[0], xlim[1], prec)
            printparamsvario(variomodelparams, variomodel, (200*i, 1.5), 7)
            plt.plot(lags2,variofunc(variomodelparams, lags2), "r", linewidth=0.5)
        if xlim:
            plt.xlim(xlim)
    plt.legend()
    plt.subplot(212)
    width=5
    for i, lenlag in enumerate(lenlags):
        icol = i%len(colors)
        plt.bar(lags+i*width, lenlags[i], color=colors[icol], width=width)
    if xlim:
        plt.xlim(xlim)
    if namefile :
        save_fig(namefile, "_anglesvarios"+str(angldev)+".png")

def plottrendxy(data):
    """
    Plot the data points with regard to the x and y axis to show a possible trend
    in the data.
    """
    indx = np.argsort(data[:,0])
    indy = np.argsort(data[:,1])
    fig, ax = plt.subplots()
    plt.subplot(211)
    plt.scatter(data[indx,0], data[indx,2])
    plt.subplot(212)
    plt.scatter(data[indy,1], data[indy,2])

def laghistogram(data, namefile, pwdist, lags, tol, width = 100):
    '''
    Input:  (data)    NumPy array with three columns, the first two
                      columns should be the x and y coordinates, and
                      third should be the measurements of the variable
                      of interest
            (pwdist)  the pairwise distances
            (lags)    the lagged distance of interest
            (tol)     the allowable tolerance about (lag)
    Output:           lag histogram figure showing the number of
                      distances at each lag
    Source : geostatsmodels.geoplot.py : https://github.com/cjohnson318/geostatsmodels
    '''
    # collect the distances at each lag
    indices = [variograms.lagindices(pwdist, lag, tol) for lag in lags]
    # record the number of indices at each lag
    indices = [len(i) for i in indices]
    # create a bar plot
    fig, ax = plt.subplots()
    ax.bar(lags + tol, indices,width)
    ax.set_ylabel('Number of Lags')
    ax.set_xlabel('Lag Distance')
    ax.set_title('Lag Histogram')
    if namefile :
        save_fig(namefile, "_laghistogram.png")
