import cell_mdl
reload(cell_mdl)
from enthought.mayavi import mlab

model = cell_mdl.Red3(Nx=50,Ny=50,Nz=0)
#model = cell_mdl.Red3(Nx=20,Ny=30)


tmdl = cell_mdl.IntPara(model)
tmdl2 = cell_mdl.IntSerial(model)
[t,v]=tmdl2.compute(tmax=100,stimCoord=[25,30,25,30],stimCoord2=[10,12,10,12])

#mlab.contour3d(v[...,-1])

#s = mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(v[...,-1]),
#                            plane_orientation='x_axes',
#                            slice_index=10,
#                            vmin = v[...,-1].min(),
#                            vmax = -30
#                        )

#s2 = mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(v[...,-1]),
#                            plane_orientation='y_axes',
#                            slice_index=10,
#                        )
#mlab.scalarbar(s,orientation='vertical',nb_labels=4,label_fmt='%.3f')
#mlab.outline()

##pour animation
#for i in range(v.shape[-1]):
#    s.mlab_source.scalars = v[...,i]
#    s2.mlab_source.scalars = v[...,i]


#logY=open('save.npz','w')
#numpy.savez(logY,t=t,v=v)
#logY.close()
