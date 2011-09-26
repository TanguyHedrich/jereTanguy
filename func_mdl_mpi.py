# Avant de lancer ce script, il faut lancer les processus qui tourneront en parallele :
#
# $ ipcluster start --n=6 --profile=mpi --engines=MPIExecEngineSetLauncher
#

def myfunc(tmax):
    from mpi4py import MPI
    import cell_mdl
    from IPython.parallel import Client
    import cPickle
    import numpy as np

    def find(f, seq):
        """fonction qui cherche une valeur dans une liste"""
        comp = 0
        for item in seq:
	        if item == f: 
		        return comp
	        comp += 1

    #recherche les processus "engine"
    rc = Client(profile='mpi')
    #Cree une vue de tous les process
    view = rc[:]

    #nombre de clients
    nCl = len(rc.ids)
    #diviseurs
    div = [i for i in range(1,nCl+1) if nCl%i==0]
    ldiv = len(div)

    if ldiv %2 == 0:
        nbx = div[ldiv/2]
        nby = div[ldiv/2-1]
    else:
        nbx = nby = div[ldiv/2]
        

    view.activate()

    #    view['tmax'] = tmax
    #    view['nbx'] = nbx
    #    view['nby'] = nby



    @view.remote(block=True)
    def parallelcomp(tmax,nbx,nby):
     
        from mpi4py import MPI
        import cell_mdl
        import numpy

        def findlimitsx(rank,nbx,Nx):
	        newNx = round( (Nx + (nbx-1) * 2) / (nbx) )
	        x = [0,0]
	        if (rank%nbx==0):
		        x = [0,newNx]
	        elif rank%nbx==(nbx-1):
		        x = [rank%nbx * newNx - rank%nbx,Nx]
		        newNx = x[1] - x[0]
	        else:
		        x[0] = rank%nbx * newNx - rank%nbx
		        x[1] = x[0] + newNx

	        return x,newNx

        def findlimitsy(rank,nby,Ny,nbx):
	        newNy = round( (Ny + (nby-1) * 2) / (nby) )
	        y = [0,0]
	        if (rank/nbx==0):
		        y = [0,newNy]
	        elif (rank/nbx==(nby-1)):
		        y = [rank/nbx * newNy - rank/nbx,Ny]
		        newNy = y[1] - y[0]
	        else:
		        y[0] = rank/nbx * newNy - rank/nbx
		        y[1] = y[0] + newNy

	        return y,newNy

#    dimension du plan
        Nx=80
        Ny=120

        rank = MPI.COMM_WORLD.Get_rank()
        # Separation de l'axe x
        [x,newNx] = findlimitsx(rank,nbx,Nx+4)
        [y,newNy] = findlimitsy(rank,nby,Ny+4,nbx)


        xyIstim1 = [3,42,4,6] 
        xyIstim2 = [40,82,95,97]
#         Coordonnees des stimuli
        if (xyIstim1[0] > x[-1]) or (xyIstim1[1] < x[0]):
            xyIstim1[0:2] = [-1,-1]
        else:
            xyIstim1[0:2] = [max(xyIstim1[0],x[0])-x[0],min(xyIstim1[1],x[-1])-x[0]]

        if (xyIstim2[0] > x[-1]) or (xyIstim2[1] < x[0]):
            xyIstim2[0:2] = [-1,-1]
        else:
            xyIstim2[0:2] = [max(xyIstim2[0],x[0])-x[0],min(xyIstim2[1],x[-1])-x[0]]


        if (xyIstim1[2] > y[-1]) or (xyIstim1[3] < y[0]):
            xyIstim1[2:4] = [-1,-1]
        else:
            xyIstim1[2:4] = [max(xyIstim1[2],y[0])-y[0],min(xyIstim1[3],y[-1])-y[0]]

        if (xyIstim2[2] > y[-1]) or (xyIstim2[3] < y[0]):
            xyIstim2[2:4] = [-1,-1]
        else:
            xyIstim2[2:4] = [max(xyIstim2[2],y[0])-y[0],min(xyIstim2[3],y[-1])-y[0]]


        mpi=[(rank%nbx==0),rank%nbx==(nbx-1),(rank/nbx==0),(rank/nbx==(nby-1))]
        m2=cell_mdl.Red3(newNx-2*(rank%nbx==0)-2*(rank%nbx==(nbx-1)),newNy-2*(rank/nbx==0)-2*(rank/nbx==(nby-1)),mpi=mpi)

        decim=20
        NbIter=0
        dt=0.05
        Ft = 0.15
        dtMin = dt
        dtMax = 6
        dVmax = 1
        thresh = 0.01
        compt = 0

        time=numpy.zeros(round(tmax/(dt*decim))+1)
        Vm=numpy.zeros((m2.Nx,m2.Ny,round(tmax/(dt*decim))+1))
    

        while (m2.time<tmax):
            Ist=0.2/2*(numpy.sign(numpy.sin(2*numpy.pi*m2.time/(1*tmax)))+1)*numpy.sin(2*numpy.pi*m2.time/(1*tmax))

            if xyIstim1[0] != -1:
                m2.Istim[xyIstim1[0]:xyIstim1[1],xyIstim1[2]:xyIstim1[3]]=Ist
            if xyIstim2[0] != -1:
                m2.Istim[xyIstim2[0]:xyIstim2[1],xyIstim2[2]:xyIstim2[3]]=Ist

            m2.derivT(dt)

        #    #Communication x
            to_send_1 = m2.Y[1,:,0]
            to_send_2 = m2.Y[-2,:,0]
            to_recv_1 = m2.Y[-1,:,0]
            to_recv_2 = m2.Y[0,:,0]

            if rank % 2:
                if rank%nbx != 0:
	                MPI.COMM_WORLD.send(to_send_1 , dest=rank-1)
                if rank%nbx != nbx - 1:
	                to_recv_1 = MPI.COMM_WORLD.recv(source=rank+1)
	                MPI.COMM_WORLD.send(to_send_2, dest=rank+1)
                if rank%nbx != 0:
	                to_recv_2 = MPI.COMM_WORLD.recv(source=rank-1)
            else:
                if rank%nbx != nbx - 1:
	                to_recv_1 = MPI.COMM_WORLD.recv(source=rank+1)
                if rank%nbx != 0:
	                MPI.COMM_WORLD.send(to_send_1 , dest=rank-1)
	                to_recv_2 = MPI.COMM_WORLD.recv(source=rank-1)
                if rank%nbx != nbx - 1:
	                MPI.COMM_WORLD.send(to_send_2, dest=rank+1)	

            m2.Y[-1,:,0] = to_recv_1 
            m2.Y[0,:,0] =  to_recv_2

            #Communication x
            to_send_1 = m2.Y[:,1,0]
            to_send_2 = m2.Y[:,-2,0]
            to_recv_1 = m2.Y[:,-1,0]
            to_recv_2 = m2.Y[:,0,0]

            if rank % 2:
                if rank/nbx != 0:
                    to_recv_2 = MPI.COMM_WORLD.recv(source=rank-nbx)
                if rank/nbx != nby - 1:
                    MPI.COMM_WORLD.send(to_send_2, dest=rank+nbx)
                    to_recv_1 = MPI.COMM_WORLD.recv(source=rank+nbx)
                if rank/nbx != 0:
                    MPI.COMM_WORLD.send(to_send_1 , dest=rank-nbx)
            else:
                if rank/nbx != nby - 1:
                    MPI.COMM_WORLD.send(to_send_2, dest=rank+nbx)
                if rank/nbx != 0:
                    to_recv_2 = MPI.COMM_WORLD.recv(source=rank-nbx)
                    MPI.COMM_WORLD.send(to_send_1 , dest=rank-nbx)	    
                if rank/nbx != nby - 1:
                    to_recv_1 = MPI.COMM_WORLD.recv(source=rank+nbx)

            m2.Y[:,-1,0] = to_recv_1
            m2.Y[:,0,0] =  to_recv_2

            m2.time =m2.time+dt
            if not round(m2.time/dt)%decim:
                NbIter+=1
                time[NbIter]=m2.time
                Vm[:,:,NbIter]=m2.Y[:,:,0].copy()
        return Nx,Ny,rank,time,x,y,Vm

    tabResults  =parallelcomp(tmax,nbx,nby)

    tabrank = np.empty(len(tabResults))

    Nx = tabResults[0][0]
    Ny = tabResults[0][1]
    time = tabResults[0][3]
    for i in range(len(tabResults)):
        tabrank[i] = tabResults[i][2]



    # Concatenation des resultats
    sol_total = np.empty((Nx+4,Ny+4,len(time)))
    for i in rc.ids:
        i_client = find(i, tabrank)
        x = tabResults[i_client][4]
        y = tabResults[i_client][5]
        sol_total[x[0]:x[1],y[0]:y[1],:]=np.array(tabResults[i_client][6])

    return time[:time.nonzero()[-1][-1]+1],sol_total[:time.nonzero()[-1][-1]+1]


