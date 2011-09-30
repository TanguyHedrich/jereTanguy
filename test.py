import cell_mdl
reload(cell_mdl)

model = cell_mdl.Red3(Nx=80,Ny=120)

tmdl = cell_mdl.IntPara(model)

[t,v] = tmdl.compute(tmax=5,stimCoord=[3,42,4,6],stimCoord2=[40,82,95,97])

tmdl.save('toto.npz')
