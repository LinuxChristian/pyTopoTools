import pyTopoTools as tt
import numpy as np

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
