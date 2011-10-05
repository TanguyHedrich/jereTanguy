import cell_mdl
from enthought.mayavi import mlab

model = cell_mdl.Red3(Nx=20,Ny=30,Nz=5)
#model = cell_mdl.Red3(Nx=20,Ny=30)


tmdl = cell_mdl.IntPara(model)
tmdl2 = cell_mdl.IntSerial(model)
[t,v]=tmdl.compute(tmax=5,stimCoord2=[10,12,10,12,2,3])
#[t,v] = tmdl.compute(tmax=250,stimCoord=[3,20,1,3])#,stimCoord2=[3,20,25,29])

#tmdl.save('toto.npz')
#mlab.surf(v)
#mlab.surf(v[...,-1],warp_scale='auto')
#mlab.show()
#mlab.contour3d(v[...,-1])
#mlab.show()
#tmdl.show()


