import cell_mdl
reload(cell_mdl)

tmdl = cell_mdl.temporal_mdl(mpi=True)

tmdl.compute(tmax=500,Nx=80,Ny=120,Nz=0,stimCoord=[3,42,4,6],stimCoord2=[40,82,95,97])

logY=open('Yv%d-%d_1P.npz'% (80,120),'w')
np.savez(logY,t=tmdl.t,Y=tmdl.Vm)
logY.close()
