# Python package to perform quick analysis on topographic data
# Copyright (C) 2015 Christian Braedstrup
# This file is part of pyTopoTools.
#
# pyTopoTools is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License v2 as published by
# the Free Software Foundation.
#
# pyTopoTools is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with pyTopoTools. If not, see <http://www.gnu.org/licenses/>.

import numpy as np
#import psd
import gdal
import pdb
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
import operator
from scipy.io import loadmat
from matplotlib.colors import LogNorm
#from mpl_toolkits.mplot3d.axes3d import Axes3D
#import parallelTopoTools as ptt
import statsmodels.api as sm
import seaborn as sns
import copy

def detrend(M):
    ny, nx = M.shape

    X, Y = np.meshgrid(range(nx),range(ny))
    A = np.vstack([np.ones(nx*ny) ,X.flatten(), Y.flatten()])

    # Compute lstsq fit to plane
    coeff, resid,rank,sigma = np.linalg.lstsq(A.T,(M.flatten()))

    # at each (x,y) point, subtract the value of the fitted plane from M
    # Zp = a + b*Y + c*X
    P = (coeff[0] + coeff[2]*Y + coeff[1]*X)
    D = M - P
    return D, P

def Hann2D(M):
    ny, nx = M.shape
    a = (nx+1)/2.0
    b = (ny+1)/2.0  # matrix coordinates of centroid of M
    [X, Y] = np.meshgrid(range(nx), range(ny))

    theta = (X==a)*(np.pi/2)
    theta += (X!=a)*np.arctan2((Y-b),(X-a))  # angular polar coordinate

    r = np.sqrt(pow2(Y-b) + pow2(X-a))  # radial polar coordinate
    r1 = pow2(b)*pow2(np.cos(theta)) + pow2(a)*pow2(np.sin(theta))
    r2 = pow2(a)*pow2(b)*np.power(r1,-1)
    rprime = np.sqrt(r2) # 'radius' of ellipse for this theta
    hanncoeff = (r < rprime)*(0.5*(1 + np.cos(np.pi*r/rprime)));
    H = M*hanncoeff;

    Wss = np.sum(np.sum(pow2(hanncoeff)));

    return [H, Wss]


def filterDEM(Z, dx, f, filttype):
    '''
    Filter the matrix Z and return
    '''
    
    Ny, Nx = Z.shape
    Zp = np.zeros((Nx,Nx))
    Zp[0:Ny,0:Nx] = Z
    Pm, fm, Pv, fv = powerspectrum(Zp,dx,pad=0,window=0)
    F = zfilter(fm,f,filttype)
    
    ft = np.fft.fftshift(np.fft.fft2(Zp))
    Zf = np.real(np.fft.ifft2(np.fft.ifftshift(ft*F)))

    return Zf[0:Ny,0:Nx], F

def detectLowRelief(Z,wstep=5,lrlim=500.0,elelim=1000.0):
    '''
    Given a topography matrix this function returns
    a binary map of low relief surfaces at high elevation

    input:
    ---------
    Z: 2D Topography matrix
    wstep: Steps of box width to compute (i.e. a list of widths to compute)
    lrlim: Limit for what a low relief is
    elelim: Cut off level for when surfaces become too low in elevation

    output:
    ---------
    Zbin: Binary low relief matrix
    '''

    Zbin = np.zeros(Z.shape)
    
    for w in wstep:
        print(w)
        ZLoc = localRelief2D(Z,w)

        Zbin[ZLoc < lrlim] += 1

        
    Zbin[Z < elelim] = 0    
    return Zbin

def plotLowRelief(Z,ZLowRe,boxsize,ext=None,cmap=None,mlab=False,fax=None):
    '''
    Plots the results of detectLowRelief
    
    INPUT:
    ------
    Z: 2D Topography matrix
    ZLowRe: Output from detectLowRelief
    boxsize: Size of boxes used in meters
    Z0: Initial Topography matix (optional)
    ZLow0: Initial low relief matrix
    ext: extent of model
    cmap: Colormap
    mlab: 3D plotting with mayavi
    fig: Figure handle
    '''

    if ext is None:
        Ny, Nx = Z.shape
        ext = [0, Ny, Nx, 0]

        
    if cmap is None:
        cmap=plt.get_cmap('jet')
        
    if mlab:
        print("Still no 3D function")
    else:
        if fax is None:
            fig = plt.figure()
            fax = fig.add_subplot(111)
            
        fax.matshow(hillshade(Z,315,65),extent=ext,cmap=plt.get_cmap('bone'))
        im_bed = fax.imshow(Z,extent=ext,cmap=cmap_center_adjust(plt.get_cmap('terrain'), 0.65),alpha=0.8)
        z_masked = np.ma.masked_where(ZLowRe < 1 , ZLowRe)
        plt.hold(True)

        plt.title('Geophysical relief')

        im_gr = fax.imshow(z_masked,extent=ext,cmap=cmap,vmin=1.0,vmax=len(boxsize))
        if 'fig' in locals():
            cax = fig.colorbar(im_gr,orientation='horizontal')
            
            cax.set_ticks(np.arange(len(boxsize)+1))
            cax.set_ticklabels(boxsize)
            plt.show()
            

def zfilter(fmat, f, filttype):
    if (filttype is 'lowpass'):
        flo = f[0]; fhi = f[1];
        mu=flo;
        sigma=np.abs(fhi-flo)/3;
        F=Gaussian(fmat,mu,sigma);
        F[fmat<flo]=1;        
    elif (filttype is 'highpass'):
        flo = f[0]; fhi = f[1];
        mu=fhi;
        sigma=np.abs(fhi-flo)/3;
        F=Gaussian(fmat,mu,sigma);
        F[fmat>=fhi]=1;        
    elif (filttype is 'bandpass'):
        flo1 = f[0]; flo2 = f[1];
        fhi1 = f[2]; fhi2 = f[3];        
        sigmalo = np.abs(flo2-flo1)/3;
        sigmahi = np.abs(fhi2-fhi1)/3;
        mulo=flo2;
        muhi=fhi1;
        Flo=Gaussian(fmat,mulo,sigmalo);
        Fhi=Gaussian(fmat,muhi,sigmahi);
        F = Flo * [fmat<=mulo] + Fhi *(fmat>=muhi) + 1*(fmat>mulo and fmat<muhi);
    elif (filttype is 'orientation'):
        # F is the radial frequency matrix
        # Slice away frequencies between f[0] and
        # f
        Ny, Nx = fmat.shape
        x = np.linspace(0,1,Nx/2)
        y = np.linspace(0,1,Ny/2)
        X,Y = np.meshgrid(x,y)
        theta = np.zeros(fmat.shape)
        theta[:Ny/2,Nx/2:Nx]   = np.rad2deg(np.arctan(np.rot90(Y/X))) # 0 - 90
        theta[Ny/2:Ny,Nx/2:Nx] = np.rad2deg(np.arctan(Y/X))+90.0 # 90 - 180
        theta[Ny/2:Ny,:Nx/2]   = np.rad2deg(np.arctan(np.rot90(Y/X,3)))+180.0 # 180 - 270
        theta[:Ny/2,:Nx/2]   = np.rad2deg(np.arctan(np.rot90(Y/X,2)))+270.0 # 270 - 360
        F = np.zeros(fmat.shape)
        F[np.where(np.logical_and(theta>=f[0],theta<=f[1]))] = 1.0
        F[np.where(np.logical_and(theta>=f[0]+180.0,theta<=f[1]+180.0))] = 1.0

    return F

# Color adjust code found on this page
# https://sites.google.com/site/theodoregoetz/notes/matplotlib_colormapadjust
def cmap_powerlaw_adjust(cmap, a):
    '''
    returns a new colormap based on the one given
    but adjusted via power-law:

    newcmap = oldcmap**a
    '''
    if a < 0.:
        return cmap
    cdict = copy.copy(cmap._segmentdata)
    fn = lambda x : (x[0]**a, x[1], x[2])
    for key in ('red','green','blue'):
        cdict[key] = map(fn, cdict[key])
        cdict[key].sort()
        assert (cdict[key][0]<0 or cdict[key][-1]>1), \
            "Resulting indices extend out of the [0, 1] segment."
    return colors.LinearSegmentedColormap('colormap',cdict,1024)

def cmap_center_adjust(cmap, center_ratio):
    '''
    returns a new colormap based on the one given
    but adjusted so that the old center point higher
    (>0.5) or lower (<0.5)
    '''
    if not (0. < center_ratio) & (center_ratio < 1.):
        return cmap
    a = np.log(center_ratio) / np.log(0.5)
    return cmap_powerlaw_adjust(cmap, a)

def cmap_center_point_adjust(cmap, range, center):
    '''
    converts center to a ratio between 0 and 1 of the
    range given and calls cmap_center_adjust(). returns
    a new adjusted colormap accordingly
    '''
    if not ((range[0] < center) and (center < range[1])):
        return cmap
    return cmap_center_adjust(cmap,
        abs(center - range[0]) / abs(range[1] - range[0]))



def localReliefBand(Z,cent,width,dim=0):
    '''
    Computes the local relief along a row/column in the
    matrix.

    input:
    ---------
    Z: 2D Topography matrix
    cent: Center of the band (pixel)
    width: Width/2 of the band (pixel)
    dim: Dimension to compute along (0/1)

    output:
    ---------
    bmin: minimum along the band
    bmax: maximum along the band
    bmean: mean along the band
    blr: local relief along the band
    '''

    if dim:
        # Compute along second dimension in matrix
        bmax = np.max(Z[:,cent-width:cent+width],axis=dim)
        bmin = np.min(Z[:,cent-width:cent+width],axis=dim)
        bmean = np.mean(Z[:,cent-width:cent+width],axis=dim)
        blr = bmax-bmin

    else:
        # Compute along first dimension in matrix
        bmax = np.max(Z[cent-width:cent+width,:],axis=dim)
        bmin = np.min(Z[cent-width:cent+width,:],axis=dim)
        bmean = np.mean(Z[cent-width:cent+width,:],axis=dim)
        blr = bmax-bmin
        
    return bmin,bmax,bmean,blr

def test_localRelief():
    Z = np.matrix([
        [1,1,1,1,1,1,1,1,1],
        [1,2,2,2,2,2,2,2,1],
        [1,2,3,3,3,3,3,2,1],
        [1,2,3,4,4,4,3,2,1],
        [1,2,3,4,5,4,3,2,1],
        [1,2,3,4,4,4,3,2,1],
        [1,2,3,3,3,3,3,2,1],
        [1,2,2,2,2,2,2,2,1],
        [1,1,1,1,1,1,1,1,1]
    ])
    
    assert 1 == localRelief(Z,[4,4],1,9,9)
    assert 2 == localRelief(Z,[4,4],2,9,9)
    assert 3 == localRelief(Z,[4,4],3,9,9)
    assert 4 == localRelief(Z,[4,4],4,9,9)
    assert 0.0 == localRelief(Z,[4,4],0,9,9)

#@profile
def localRelief(Z,c,w,Nx,Ny):
    '''
    Given a center point in pixel and a box width this function
    computes the local relief with that rectangle.
    Notice that w is the distance in each direction. The
    box width is therefore 2*w. (i.e. w=1 is the area within 1 pixel
    from the center c).

    input:
    -------------
    Z: Topography matrix
    c: (x,y) of center point in pixel
    w: Width of area to compute local relief within (halv box width)
    Nx: Number of cells in x
    Ny: Number of cells in y
    '''

    # Boundary conditions
    xl = c[0]-w if c[0]-w > 0 else 0
    xr = c[0]+w if c[0]+w < Nx else Nx
    yl = c[1]-w if c[1]-w > 0 else 0
    yr = c[1]+w if c[1]+w < Ny else Ny

    sli = Z[yl:yr,xl:xr]
    if (len(sli) > 0):
#    return ptt.mima(sli)
#    return np.max(sli)-np.min(sli)
        return np.amax(sli)-np.amin(sli)
    else:
        return 0.0

    
def test_localRelief2D():
    Z = np.matrix([
        [1,1,1,1,1,1,1,1,1],
        [1,2,2,2,2,2,2,2,1],
        [1,2,3,3,3,3,3,2,1],
        [1,2,3,4,4,4,3,2,1],
        [1,2,3,4,5,4,3,2,1],
        [1,2,3,4,4,4,3,2,1],
        [1,2,3,3,3,3,3,2,1],
        [1,2,2,2,2,2,2,2,1],
        [1,1,1,1,1,1,1,1,1]
    ])

    b = localRelief2D(Z,4)
    assert 4 == b[4,4]

def localRelief2D(Z,width=5,walti=False):
    '''
    Computes the local relief using a window function
    with default width of 5 px in each direction. 
    The window function is compute by the function localRelief.

    input:
    -------------
    Z: Topography matrix
    width: Width of area to compute local relief within
    walti: Compute mean relief in altitude bands
    '''
    
    Ny,Nx = Z.shape
    Zloc = np.ones(Z.shape)*1e4     # Start with a high local relief everywhere
    d = width
#    print(Ny,Nx)
    for x in np.linspace(d,Nx-d,Nx-2*d):
        for y in np.linspace(d,Ny-d,Ny-2*d):
            Zloc[y,x] = localRelief(Z,[x,y],d,Nx,Ny)

    
    # Group relief into altitude bands of 50 meters    
    if walti:
        gmin = np.amin(Z)
        gmin = gmin-np.mod(gmin,100)+300.0
        gmax = np.amax(Z)
        gmax = gmax-np.mod(gmax,100)-100.0
        rrange = np.arange(gmin,gmax,50.0)
        Zele = np.zeros( (len(rrange),2) )  # Matrix to hold elevatio and mean local relief

        # Only work with even 100 numbers
        for i in range(len(rrange)):
            if i == 0:
                Zele[0,0] = rrange[0]
                Zele[0,1] = np.mean(Zloc[Z < rrange[0]])
            elif i == len(rrange)-1:
                Zele[-1,0] = rrange[-1]
                Zele[-1,1] = np.mean(Zloc[Z > rrange[-1]])
            else:
                Zele[i,0] = rrange[i]
                Zele[i,1] = np.mean(Zloc[np.where(np.logical_and(Z>rrange[i],Z<rrange[i+1]))])
                
        return Zloc, Zele

    else:
        return Zloc
            


def test_boxRelief():
    Z = np.matrix([
        [1,1,1,1,1,1,1,1,1],
        [1,2,2,2,2,2,2,2,1],
        [1,2,3,3,3,3,3,2,1],
        [1,2,3,4,4,4,3,2,1],
        [1,2,3,4,5,4,3,2,1],
        [1,2,3,4,4,4,3,2,1],
        [1,2,3,3,3,3,3,2,1],
        [1,2,2,2,2,2,2,2,1],
        [1,1,1,1,1,1,1,1,1]
    ])

    assert 4 == boxRelief(Z,1,1,[4,4],4,rn=2,evenly=True)

def boxRelief(Z,dx,dy,c,w=5,rn=10,evenly=False,plot=False):
    '''
    Computes relief within a box of 
    width w (cells) with rn intervals
    centered around c [xc,yc].

    INPUT
    ------
    Z: 2D Topography matrix
    dx: X spacing
    dy: Y spacing
    c: Box center point
    w: Box width
    rn: Number of box increases
    evenly: Grow box evenly or just in one direction
    plot: Plot result
    '''

    Nx,Ny = Z.shape
    L = dx*Nx; H = dy*Ny;
    lrelief = np.zeros((rn-1))

    # Evenly distribute
    if evenly is True:
        for i in np.arange(1,rn):
            lrelief[i-1] = localRelief(Z,c,w*i,Nx,Ny)
    else:
        for i in np.arange(1,rn):
            # Boundary conditions
            xl = c[0] if w > 0 else c[0]+i*w
            xr = c[0]+i*w if w > 0 else c[0]
            yl = c[1] if w > 0 else c[1]+i*w
            yr = c[1]+i*w if w > 0 else c[1]

            sli = Z[yl:yr,xl:xr]
            if (len(sli) > 0):
                lrelief[i-1] = np.amax(sli)-np.amin(sli)

    if plot:
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        plt.plot(np.arange(1,rn)*np.sqrt(dx*dx+dy*dy),lrelief,'-*')
        plt.title('Box relief')
        plt.ylabel('Relief')
        plt.xlabel('Box width (m)')
        
        ax2 = fig.add_subplot(212)
        plt.imshow(Z)
        plt.hold(True)
        for i in np.arange(1,rn,2):
            ax2.add_patch(patches.Rectangle(c,i*w,i*w,fill=None,color='k'))
        plt.colorbar()
        plt.show()

    print(lrelief)
    return lrelief

def bslope(Z,dx=1,dy=1,cmin=0.0,cmax=1e10):
    '''
    Computes the local bed slope as
    the laplacian for the bed.
    '''

    Zgx, Zgy = np.gradient(Z,dx,dy)

    if cmin != 0.0 or cmax != 1e10:
        return np.clip(np.sqrt(pow2(Zgx)+pow2(Zgy)),cmin,cmax)
    else:
        return np.sqrt(pow2(Zgx)+pow2(Zgy))

def analyseDEM(Z,dx,dy,forceshow=False,w=5,title='',trans=False,corner=[0.0,0.0],divcmap=None,seqcmap=None):
    '''
    Wrapper function to provide a quick
    overview of a DEM.
    Plots shadedrelief, local relief and
    bed gradient.

    INPUT:
    --------
    Z: 2D Topography Matrix
    dx: Cell width in x-dimension
    dy: Cell width in y-dimension
    forceshow: Show plot after function call
    w: Box width for geophysical relief
    title: Super title for plot
    trans: Transpose Z matrix for plotting
    corner: Coordinates of (x0,y0)
    cmap: custom colormap
    '''
    if trans:
        Z = Z.T

    Ny, Nx = Z.shape
    x0 = corner[0]
    x1 = x0 + Nx*dx
    x1 = x1/1e3 # to km
    y0 = corner[1]
    y1 = y0 + Ny*dy
    y1 = y1/1e3 # to km

    # Use custom Seaborn color palettes
    if divcmap is None:
        # Diverging cmap
        divcmap = sns.diverging_palette(220, 20, n=7, as_cmap=True)
        
    if seqcmap is None:
        # Sequential cmap
        seqcmap = sns.light_palette("green", reverse=True, as_cmap=True)
    
    plt.figure()
    plt.subplot(221)
    plt.imshow(Z,cmap=plt.get_cmap('terrain'),extent=[y0, y1, x1, x0],aspect='auto')
    plt.title("Elevation",fontsize=12)
    plt.ylabel('Km')
    plt.colorbar(orientation='horizontal')
    
    plt.subplot(222)
    Zloc = localRelief2D(Z,w)
    plt.imshow(Zloc,vmax=150,extent=[y0, y1, x1, x0],aspect='auto',cmap=seqcmap)
    plt.title("Local relief - Radius %i"%(np.sqrt(pow2(w*dx)+pow2(w*dy))))
    plt.ylabel('Km')
    plt.colorbar(orientation='horizontal')
    
    plt.subplot(223)
    Zslo = bslope(Z,dx,dy,cmax=1.0)
    plt.imshow(Zslo,vmax=0.3,extent=[y0, y1, x1, x0],aspect='auto',cmap=divcmap)
    plt.title("Gradient")
    plt.ylabel('Km')
    plt.colorbar(orientation='horizontal')

    plt.subplot(224)
    ZlocZ = np.clip(Zloc/Z,0.0,1)
    plt.imshow(ZlocZ,extent=[y0, y1, x1, x0],aspect='auto',cmap=divcmap)
    plt.ylabel('Km')
    plt.title("Local Relief/Elevation")
    plt.colorbar(orientation='horizontal')

    plt.tight_layout()
    
    if forceshow:
        plt.show()

    return Zloc, Zslo, ZlocZ


def PowerspecOverview(Z,dx,dy=0):
    '''
    Wrapper function to give a quick overview
    of the topography
    '''

    Z, P = detrend(Z)
    Pm0, fm0, Pv0, fv0 = powerspectrum(Z,dx,pad=1,window=1)
    xc, Bp, ppoly = denoise(fv0,Pv0,remove=3)
    ffit = ppoly.coeffs # c + a*x, [a,c]
    print(ffit)
    # Plot results
    
    plt.figure()
    ax1 = plt.subplot(211)
    plt.loglog(fv0,Pv0,'ko')
    plt.hold(True)
    plt.loglog(np.power(10,xc),np.power(10,Bp),'ro')
#    plt.plot(xc,(np.power(10,ffit[0])*np.power(xc,ffit[1])),'r-')
    plt.plot(np.power(10,xc[3:]),np.power(10,ffit[0]*xc[3:]+ffit[1]),'r-')

#    xrang = ax1.get_xticks()
#    x2rang = np.divide(1.0,xrang)
#    ax2 = ax1.twiny()
#    print(x2rang)
#    ax1.set_xticks(xrang,str(x2rang))
    ax1.set_xlabel('frequency (1/m)', color='k')
#    ax2.set_xlim(x2rang)
    ax1.set_xscale('log')
    plt.ylabel('Mean-squared amplitude (m^2)')
    plt.title('Full 1D Powerspectrum')
    
    ax1 = plt.subplot(212)
    Pmm = Pm0/(np.power(10,ffit[1])*np.power(fm0,ffit[0]))
    plt.plot(fm0,Pmm,'k*')
    ax1.set_xscale('log')

    plt.title('Filtered 1D Powerspectrum')
    plt.xlabel('frequency (1/m)')
    plt.ylabel('Normalized power')
    plt.tight_layout()
    
    plt.figure()
    # [400:600,400:600]
    plt.imshow(np.log10(Pmm),norm=LogNorm(0.1,1))
    plt.colorbar()
    plt.title('2D Powerspectrum')

    Nx, Ny = Pm0.shape
    L = Nx*dx
    H = Ny*dy
    nq = 1/(2*Nx)
    plt.xticks(np.linspace(0,Nx,10),np.linspace(-nq,nq,10))
    plt.show()


def plotDEM(Z,hshade=False,t=''):
    if (hshade is True):
        plt.matshow(hillshade(Z,315,45),cmap='Greys')
    else:
        plt.matshow(Z)
        plt.title(t)
        plt.colorbar()
        
    plt.show()

def plotVariable(varname,cmap=None,trans=False,title=''):
    '''
    Plot the variable from SPM output files

    INPUT:
    -------
    var: variable to plot
    clim: color limits
    '''
    if cmap is None:
        cmap = sns.cubehelix_palette(8,as_cmap=True)

    if trans:
        varname = varname.T
        
    plt.figure()
    ax = plt.imshow(varname,cmap=cmap)
    plt.title(title)
    cbar = plt.colorbar(ax)

def outputGeoTiff(Z,outfile,trans,dim,prj=None):
    """
    Outputs a matrix Z as a geotiff file.

    INPUT
    ---------
    Z: 2D Topography Matrix
    outfile: Path to output file
    trans: Transform matrix [x0, dx, 0.0, 0.0, y0, dy]
    dim: [Nx,Ny]
    prj: Projection infomation
    """

    output_raster = gdal.GetDriverByName('GTiff').Create(outfile, dim[0], dim[1], 1 ,gdal.GDT_Float32)
    output_raster.SetGeoTransform(trans)  # Specify its coordinates
    if prj is not None:
        srs = osr.SpatialReference(wkt=prj)                 # Establish its coordinate encoding
        output_raster.SetProjection( srs.ExportToWkt() )   # Exports the coordinate system to the file
    output_raster.GetRasterBand(1).WriteArray(Z)   # Writes my array to the raster
    output_raster = None

def hillshade(array, azimuth, angle_altitude):
        
    x, y = np.gradient(array)
    slope = np.pi/2. - np.arctan(np.sqrt(x*x + y*y))
    aspect = np.arctan2(-x, y)
    azimuthrad = azimuth*np.pi / 180.
    altituderad = angle_altitude*np.pi / 180.
     
 
    shaded = np.sin(altituderad) * np.sin(slope)\
     + np.cos(altituderad) * np.cos(slope)\
     * np.cos(azimuthrad - aspect)
    return 255*(shaded + 1)/2

def denoise(fv,Pv,remove=0):
    '''
    Remove noise by using a power-law fit
    '''

    fv10 = np.log10(fv)
    nbin = 20; # Number of bins

    Bf,bedge = np.histogram(fv10,bins=nbin); # Bin the log-transformed data

    # Compute bin centers
    xc = np.array([ np.mean([bedge[i],bedge[i+1]]) for i in range(len(bedge)-1)])

    Bp = np.zeros(len(Bf))
    for i in range(0,len(bedge)-1):
        if i==0:
            Bp[i] = np.mean(Pv[fv10<bedge[i]])
        elif i==len(bedge)-1:
            Bp[i] = np.mean(Pv[fv10>bedge[i]])
        else:
            Bp[i] = np.mean(Pv[np.where(np.logical_and(fv10>bedge[i-1],fv10<bedge[i]))])


    Bp = np.log10(Bp)
    Bp[np.isnan(Bp)]=0.0
    psqrt = np.polyfit(xc[remove:].flatten(),Bp[remove:].flatten(),1)

#    if (remove == -1):
#        A = np.vstack([np.zeros(nbin) ,xc.flatten()])
#    else:
#        A = np.vstack([np.zeros(nbin-remove) ,xc[remove:].flatten()])
#        print(xc[remove:].flatten())
        
    # Compute lstsq fit to line
#    coeff, resid,rank,sigma = np.linalg.lstsq(A.T,(Bp[remove:].flatten()))

    return xc, Bp, np.poly1d(psqrt)

def Gaussian(freqmat,mu,sigma):
    G=np.exp(-pow2(freqmat-mu)/(2*pow2(sigma)))
    G=G/np.amax(G)

    return G


def radial_profile(data, center):
    y, x = np.indices(data.shape)
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile 

def show(M,typ=0):
    ny, nx = M.shape
    X, Y = np.meshgrid(range(nx), range(ny))

    fig = plt.figure()
    if typ == 1:
        ax = fig.add_subplot(1,1,1,projection='3d')
        ax.plot_surface(X,Y,M)
    else:
        plt.pcolormesh(M)        
        plt.colorbar()
    plt.show()

def pow2(x):
    return np.power(x,2)

def powerspectrum(M, dx, dy=0, pad=0, window=0):
    if dy==0:
        dy=dx

    ny, nx = M.shape

    # Data padding
    if pad:
        # calculate the power of 2 to pad with zeros 
        Lx = int(np.power(2,(np.ceil(np.log(np.max([nx, ny]))/np.log(2)))))
        Ly = int(Lx)
    else:
        # no zero padding
        Lx = int(nx)
        Ly = int(ny)


    if window:
    # window the matrix with an elliptical Hann (raised cosine) window
        M, Wss = Hann2D(M)
    else:
    # do not window (really, a square window with a value of 1)
        Wss = np.sum(np.sum(np.ones((ny, nx))));        

    # calculate the frequency increments: frequency goes from zero (DC) to
    # 1/(2*dx) (Nyquist in x direction) in Lx/2 increments; analogous for y.
    dfx = 1/float(dx*Lx)
    dfy = 1/float(dy*Ly)

    M = np.rot90(np.fft.fftshift(np.fft.fft2(M,(Ly,Lx))))
    M = np.real(M * np.conj(M)) / (Lx * Ly * Wss)

    M[Ly/2+1,Lx/2+1]=0
    # assign the power spectrum to the output argument
    Pm = M

    # Create a matrix of radial frequencies
    xc = Lx/2
    yc = Ly/2
    cols, rows = np.meshgrid(range(Lx),range(Ly))

    fm = np.sqrt(pow2(dfy*(rows-yc)) + pow2(dfx*(cols-xc)))  # frequency matrix
    # Create sorted, non-redundant vectors of frequency and power 
    M = M[:,range(Lx/2+1)]
    fv = fm[:,range(Lx/2+1)]
    fv[yc:Ly,xc-1] = -1


    # Sort frequency vector and powerspec vector
    fv = fv.flatten(1)
    fvIdx = fv.argsort()  # Get sorted index
    Pv = Pm.flatten(1)
    fv = fv[fvIdx]
    Pv = Pv[fvIdx]
    # Remove negative frequencies
    Pv = Pv[fv>0]
    fv = fv[fv>0]

    # Separate into power and frequency vectors and assign to output arguments
    Pv = 2*Pv

    return [Pm,fm,Pv,fv]
