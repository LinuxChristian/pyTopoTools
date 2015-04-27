#
#
# This set of functions only apply to output files
# produced by the surface process models developed at
# the University of Aarhus, Denmark.
#
#

import pyTopoTools as tt
import numpy as np
import os

def loadSPM(ProjectPath,revision,flipxy=0, FileNo=0):
    '''
    Load output from SPM models    
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

def loopSPMFiles(ProjectPath,revision,cent,w,h,tsize,tstep=1,flipxy=0,fnum=None):

    # Find number of output files
    if fnum is None:
        fnum = os.listdir( ProjectPath+revision+'/output/' )

    trang = np.arange(1,fnum+1,tstep)
    atime = np.zeros((len(trang)))
    aero = np.zeros((len(trang)))
    aeror = np.zeros((len(trang)))
    alr = np.zeros((len(trang)))

    # Load initial topography
    L,H,Nx,Ny,dx,dy,Z0 = loadSPM(ProjectPath,revision,flipxy,0)
    
    for i,f in enumerate(trang):
        print("Loading file %i" % (f))
        L,H,Nx,Ny,dx,dy,Z = loadSPM(ProjectPath,revision,flipxy,f)
        atime[i] = f*tsize
        aero[i], aeror[i], alr[i] = areaErosion(Z,Z0,cent,w,h,atime[i])

    return atime, aero, aeror, alr

def areaErosion(Z,Zbase,cent,w,h,time):
    '''
    Computes the mean erosion and local relief generation
    within a limited area.

    input:
    --------
    Z: 2D Topography matrix
    Zbase: Initial 2D topography matrix
    cent: Center of area (x,y)
    w: Width of area
    h: Height of area
    time: Current compute time

    output:
    ---------
    Mean erosion
    Mean erosion rate
    Local relief within area
    '''
    Zarea = Z[cent[0]-w:cent[0]+w,cent[1]-h:cent[1]+h]
    Zbarea = Zbase[cent[0]-w:cent[0]+w,cent[1]-h:cent[1]+h]
    
    return np.mean(Zarea-Zbarea), np.mean(Zarea-Zbarea)/time, np.amax(Zarea)-np.amin(Zarea)
