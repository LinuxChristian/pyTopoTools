import numpy as np
#import psd
import gdal
import pdb
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import operator
from scipy.io import loadmat
from matplotlib.colors import LogNorm
#from mpl_toolkits.mplot3d.axes3d import Axes3D
import statsmodels.api as sm

def detrend(M):
    ny, nx = M.shape

    X, Y = np.meshgrid(range(nx),range(ny))
    A = np.vstack([np.ones(nx*ny) ,X.flatten(), Y.flatten()])

    # Compute lstsq fit to plane
    coeff, resid,rank,sigma = np.linalg.lstsq(A.T,(M.flatten()))

    # at each (x,y) point, subtract the value of the fitted plane from M
    # Zp = a + b*Y + c*X
    print(np.amax(M))
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

def localRelief(Z,width=5,walti=False):
    '''
    Computes the local relief using a window function
    with default width of 5 px.

    input:
    -------------
    Z: Topography matrix
    width: Width of area to compute local relief within
    walti: Compute mean relief in altitude bands
    '''
    
    Nx,Ny = Z.shape
    Zloc = np.zeros(Z.shape)
    d = np.floor(width/2)
    for x in np.linspace(d,Nx-d,Nx-2*d):
        for y in np.linspace(d,Ny-d,Ny-2*d):
            Zloc[x,y] = np.amax(Z[x-d:x+d,y-d:y+d])-np.amin(Z[x-d:x+d,y-d:y+d])

    
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
            


def boxRelief(Z,dx,dy,c,w=5,rn=10,plot=False):
    '''
    Computes relief within a box of 
    width w (cells) with rn intervals
    centered around c [xc,yc].
    '''

    Nx,Ny = Z.shape
    L = dx*Nx; H = dy*Ny;
    lrelief = np.zeros((rn-1))
    for i in np.arange(1,rn):
        # Boundary conditions
        xl = c[0] if w > 0 else c[0]+i*w
        xr = c[0]+i*w if w > 0 else c[0]
        yl = c[1] if w > 0 else c[1]+i*w
        yr = c[1]+i*w if w > 0 else c[1]
        
        sli = Z[yl:yr,xl:xr]
#        print(xl,xr,yl,yr)
#        print(sli)
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

def analyseDEM(Z,dx,dy,forceshow=False,w=5):
    '''
    Wrapper function to provide a quick
    overview of a DEM.
    Plots shadedrelief, local relief and
    bed gradient.
    '''
    plt.figure()
    plt.subplot(221)
    plt.imshow(Z)
    plt.title("Elevation")
    plt.colorbar()
    
    plt.subplot(222)
    Zloc = localRelief(Z,w)
    plt.imshow(Zloc,vmax=350)
    plt.title("Local relief - Radius %i"%(np.sqrt(pow2(w*dx)+pow2(w*dy))))
    plt.colorbar()

    plt.subplot(223)
    Zslo = bslope(Z,dx,dy,cmax=1.0)
    plt.imshow(Zslo,vmax=0.5)
    plt.title("Sognefjord - Gradient")
    plt.colorbar()

    plt.subplot(224)
    ZlocZ = np.clip(Zloc/Z,0.0,1)
    plt.imshow(ZlocZ)
    plt.title("Sognefjord - Local Relief/Elevation")
    plt.colorbar()

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


def loadSPM(ProjectPath,revision,flipxy=0, FileNo=0):
    '''
    Load output from SPM
    '''

    # Get infomation about model setup
    fid = open(ProjectPath+revision+"/input/mesh.input","r")
    L = float(fid.readline())
    H = float(fid.readline())
    nx = int(fid.readline())
    ny = int(fid.readline())

    if (flipxy == 1):
        ff = nx
        nx = ny
        ny = ff

    dx  = L/nx
    dy  = H/ny

    fid.close()

    junk  = np.zeros((nx,ny), dtype=np.float64)
    bed = np.zeros((nx,ny), dtype=np.float64)

    with open(ProjectPath+revision+"/output/output"+str(FileNo)+".dat","rb") as f:
        for y in np.arange(0, ny):            
            junk[:,y]          = np.fromfile(f,dtype=np.float64, count=nx);
        for y in np.arange(0, ny):
            junk[:,y]          = np.fromfile(f,dtype=np.float64, count=nx);
        for y in np.arange(0, ny):
            bed[:,y]         = np.fromfile(f,dtype=np.float64, count=nx);

    f.close()

    return [L,H,nx,ny,dx,dy,bed]

def plotDEM(Z,hshade=False,t=''):
    if (hshade is True):
        plt.matshow(hillshade(Z,315,45),cmap='Greys')
    else:
        plt.matshow(Z)
        plt.title(t)
        plt.colorbar()
        
    plt.show()
    
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
