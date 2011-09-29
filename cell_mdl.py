"""Tissue models classes:
Tissuemodel: Generic base class, not a functionnal model by itself.
Red3: Uses reduced 3 vars uterine cell model (J.Laforet).
Red6: Uses reduced 6 vars uterine cell model (S.Rihana)."""

import numpy
import pylab
from scipy.ndimage.filters import correlate1d
from IPython.parallel import Client

class TissueModel:
    """Generic cell and tissue model."""
    def __init__(self,dim,Nx,Ny=0,Nz=0,noise=0.0,borders=[True,True,True,True,True,True]):
        """Model init.
            dim: number of variables of state vector.
            Nx: number of cells along X.
            Nx: number of cells along X.
            Nx: number of cells along X.
            noise: noise coefficient for initial state.
            borders: boolean array [firstX,lastX,firstY,lastY,firstZ,lastZ]"""
        #dimensions
        self.Name="Generic!"
        self.Padding=4
        self.time=0
        #Initialise state given the type of model
        if dim==3:
            Y0=[-50,0.079257,0.001]
        elif dim==6:
            Y0=[-50,0.0015709,0.8,0.8,0.079257,0.001]
        else:
            Y0=numpy.zeros(dim)
        #state
        if Nx*Ny*Nz:
            #update dims with padding
            self.Nx=Nx+borders[0]*self.Padding/2+borders[1]*self.Padding/2
            self.Ny=Ny+borders[2]*self.Padding/2+borders[3]*self.Padding/2
            self.Nz=Nz+borders[4]*self.Padding/2+borders[5]*self.Padding/2
            #generate full state
            self.Y=numpy.tile(numpy.array(Y0),(self.Nx,self.Ny,self.Nz,1))
            #mask for padding borders  
            self.mask=1e-4*numpy.ones(self.Y.shape[0:-1])
            self.mask[borders[0]*self.Padding/2:self.Nx-borders[1]*self.Padding/2,
                      borders[2]*self.Padding/2:self.Ny-borders[3]*self.Padding/2,
                      borders[4]*self.Padding/2:self.Nz-borders[5]*self.Padding/2
                      ]=numpy.ones((self.Nx-borders[0]*self.Padding/2-borders[1]*self.Padding/2,
                                    self.Ny-borders[2]*self.Padding/2-borders[3]*self.Padding/2,
                                    self.Nz-borders[4]*self.Padding/2-borders[5]*self.Padding/2))
            #diffusion coeffs
            self.Dx=2.222/16
            self.Dy=2.222/16
            self.Dz=2.222/16
            self.derivS=self._derivS3
        elif Nx*Ny:
            self.Nx=Nx+borders[0]*self.Padding/2+borders[1]*self.Padding/2
            self.Ny=Ny+borders[2]*self.Padding/2+borders[3]*self.Padding/2
            self.Y=numpy.tile(numpy.array(Y0),(self.Nx,self.Ny,1))
            #mask for padding borders    
            self.mask=1e-4*numpy.ones(self.Y.shape[0:-1])
            self.mask[borders[0]*self.Padding/2:self.Nx-borders[1]*self.Padding/2,
                      borders[2]*self.Padding/2:self.Ny-borders[3]*self.Padding/2
                      ]=numpy.ones((self.Nx-borders[0]*self.Padding/2-borders[1]*self.Padding/2,
                                    self.Ny-borders[2]*self.Padding/2-borders[3]*self.Padding/2))
            #diffusion coeffs
            self.Dx=2.222/16
            self.Dy=2.222/16
            self.derivS=self._derivS2
        elif Nx>1:
            self.Nx=Nx+borders[0]*self.Padding/2+borders[1]*self.Padding/2
            self.Y=numpy.tile(numpy.array(Y0),(self.Nx,1))
            #mask for padding borders    
            self.mask=1e-4*numpy.ones(self.Y.shape[0:-1])
            self.mask[borders[0]*self.Padding/2:self.Nx-borders[1]*self.Padding/2
                      ]=numpy.ones((self.Nx-borders[0]*self.Padding/2-borders[1]*self.Padding/2))  
            #diffusion coeffs
            self.Dx=2.222/16
            self.derivS=self._derivS1                           
        else:
            self.Y=numpy.array(Y0)
            self.derivS=self._derivS0
       
        #option for noisy initial state
        if noise!=0.0:
            self.Y*=1+(numpy.random.random(self.Y.shape)-.5)*noise    
        #parameters
        self.R=8.314
        self.T=295
        self.F=96.487
        self.Ca0=3*numpy.ones(self.Y.shape[0:-1])
        self.Cm=1
        self.Ra=500
        self.h=0.03
        self.Istim=numpy.zeros(self.Y.shape[0:-1])
        self.stimCoord=[0,0,0,0]
        self.stimCoord2=[0,0,0,0]
        
    def __repr__(self):
        """Print model infos."""
        return "Model {}, dimensions: {}.".format(self.Name,self.Y.shape)   
    def _derivative2(self,inumpyut, axis, output=None, mode="reflect", cval=0.0):
        return correlate1d(inumpyut, [1, -2, 1], axis, output, mode, cval, 0)
    def diff1d(self,Var):
        Dif=self.Dx*self._derivative2(Var,0)
        Dif[self.stimCoord[0]:self.stimCoord[1]]=0
        Dif[self.stimCoord2[0]:self.stimCoord2[1]]=0
        return Dif*self.mask   
    def diff2d(self,Var):
        Dif=self.Dx*self._derivative2(Var,0)+self.Dy*self._derivative2(Var,1)
        Dif[self.stimCoord[0]:self.stimCoord[1],self.stimCoord[2]:self.stimCoord[3]]=0
        Dif[self.stimCoord2[0]:self.stimCoord2[1],self.stimCoord2[2]:self.stimCoord2[3]]=0
        return Dif*self.mask
    def diff3d(self,Var):
        Dif=self.Dx*self._derivative2(Var,0)+self.Dy*self._derivative2(Var,1)+self.Dz*self._derivative2(Var,2)
        Dif[self.stimCoord[0]:self.stimCoord[1],self.stimCoord[2]:self.stimCoord[3],0]=0
        Dif[self.stimCoord2[0]:self.stimCoord2[1],self.stimCoord2[2]:self.stimCoord2[3],0]=0
        return Dif*self.mask    
    def _derivS0(self):
        """Computes spatial derivative to get propagation."""
        pass
    def _derivS1(self):
        """Computes spatial derivative to get propagation."""
        self.dY[...,0]+=self.diff1d(self.Y[...,0])
    def _derivS2(self):
        """Computes spatial derivative to get propagation."""
        self.dY[...,0]+=self.diff2d(self.Y[...,0])
    def _derivS3(self):
        """Computes spatial derivative to get propagation."""
        self.dY[...,0]+=self.diff3d(self.Y[...,0])    
    def plotstate(self):
        """Plot state of the model with the suitable method, according its dimensions."""
        if self.Y.ndim==1:
            print "State: Vm={0} mV, nK={1} and [Ca]={2} mmol.".format(self.Y[0],self.Y[1],self.Y[2])
        elif self.Y.ndim==2:
            #pylab.figure()
            pylab.plot(self.Y)
            pylab.legend( ('Vm','nK','[Ca]'))
            #pylab.show()
        elif self.Y.ndim==3:
            #pylab.figure()
            pylab.subplot(212)
            pylab.imshow(self.Y[...,0])
            pylab.colorbar()
            pylab.title('Vm')
            pylab.subplot(221)
            pylab.imshow(self.Y[...,1])
            pylab.colorbar()
            pylab.title('nK')
            pylab.subplot(222)
            pylab.imshow(self.Y[...,2])
            pylab.colorbar()
            pylab.title('[Ca]')
            #pylab.show()
        elif self.Y.ndim==4:
            print "Display of 2.5D models currrently unsupported."
        else:
            print "Uncompatible model dimensions for plotting."

class Red3(TissueModel):
    """Cellular and tissular model Red3"""
    def __init__(self,Nx,Ny=0,Nz=0,noise=0.0,borders=[True,True,True,True,True,True]):
        """Model init."""
        #Generic elements
        TissueModel.__init__(self,3,Nx,Ny,Nz,noise,borders)
        self.Name="Red3"
        self.Gk=0.064
        self.Gkca=0.08
        self.Gl=0.0055
        self.Kd=0.01
        self.fc=0.4
        self.alpha=4*10**-5
        self.Kca=0.01
        self.El=-20
        self.Ek=-83
        self.Gca2=-0.02694061
        self.vca2=-20.07451779
        self.Rca=5.97139101
        self.Jbase=0.02397327
        self.dY=numpy.empty(self.Y.shape)
        #self.Istim[5:20,5]=0.2
        
    def derivT(self,dt):
        """Computes temporal derivative for red3 model."""
        #Variables
        Vm=self.Y[...,0]
        nk=self.Y[...,1]
        Ca=self.Y[...,2]  
        #Nerst
        Eca=((self.R*self.T)/(2*self.F))*numpy.log(self.Ca0/Ca)
        #H inf x
        hki=1/(1+numpy.exp((4.2-Vm)/21.1))
        #Tau x
        tnk=23.75*numpy.exp(-Vm/72.15)
        #Courants
        Ica2=self.Jbase-self.Gca2*(Vm-Eca)/(1+numpy.exp(-(Vm-self.vca2)/self.Rca))
        Ik=self.Gk*nk*(Vm-self.Ek)
        Ikca=self.Gkca*Ca**2/(Ca**2+self.Kd**2)*(Vm-self.Ek)
        Il=self.Gl*(Vm-self.El)        
        #Derivees
        self.dY[...,0] = (self.Istim - Ica2 -Ik - Ikca -Il)/self.Cm
        self.dY[...,1] = (hki-nk)/tnk
        self.dY[...,2] = self.fc*(-self.alpha*Ica2 - self.Kca*Ca)
        #update Y
        self.derivS()
        self.Y+=self.dY*dt
        
class Red6(TissueModel):
    """Cellular and tissular model Red6"""
    def __init__(self,Nx,Ny=0,Nz=0,noise=0.0,borders=[True,True,True,True,True,True]):
        """Model init."""
        #Generic elements
        TissueModel.__init__(self,6,Nx,Ny,Nz,noise,borders)
        self.Name="Red6"
        self.Gca=0.09
        self.Gk=0.064
        self.Gkca=0.08
        self.Gl=0.0055
        self.Kd=0.01
        self.fc=0.4
        self.alpha=4*10**-5
        self.Kca=0.01
        self.El=-20
        self.Ek=-83
        self.Gca2=-0.02694061
        self.vca2=-20.07451779
        self.Rca=5.97139101
        self.Jbase=0.02397327
        self.dY=numpy.empty(self.Y.shape)
        #self.Istim[5:20,5]=0.2
        
    def derivT(self,dt):
        """Computes temporal derivative for red3 model."""
        #Variables
        Vm=self.Y[...,0]
        mca=self.Y[...,1]
        h1ca=self.Y[...,2]
        h2ca=self.Y[...,3]
        nk=self.Y[...,4]
        Ca=self.Y[...,5]
        #Nerst
        Eca=((self.R*self.T)/(2*self.F))*numpy.log(self.Ca0/Ca)

        
        #H inf x
        mcai=1/(1+numpy.exp((-27-Vm)/6.6))
        hcai=1/(1+numpy.exp((Vm+34)/5.4))
        hki=1/(1+numpy.exp((4.2-Vm)/21.1))

        #Tau x
        tmca=0.64*numpy.exp(-0.04*Vm)+1.188
        th1ca=160*numpy.ones(Vm.shape)
        Imodif=numpy.nonzero((Vm<-10)|(Vm>45))
        th1ca[Imodif]=24.65*numpy.exp(-0.07281*Vm[Imodif])+17.64*numpy.exp(0.029*Vm[Imodif])
        th2ca=160
        tnk=23.75*numpy.exp(-Vm/72.15)
       
        #Alias
        fca=1/(1+Ca)
        hca=0.38*h1ca+0.22*h2ca+0.06

        #Courants
        Ica=self.Gca*mca*mca*hca*fca*(Vm-Eca)
        Ik=self.Gk*nk*(Vm-self.Ek)
        Ikca=self.Gkca*Ca**2/(Ca**2+self.Kd**2)*(Vm-self.Ek)
        Il=self.Gl*(Vm-self.El)
            
        #Derivees
        self.dY[...,0] = (self.Istim - Ica -Ik - Ikca -Il)/self.Cm
        self.dY[...,1] = (mcai-mca)/tmca
        self.dY[...,2] = (hcai-h1ca)/th1ca
        self.dY[...,3] = (hcai-h2ca)/th2ca
        self.dY[...,4] = (hki-nk)/tnk
        self.dY[...,5] = self.fc*(-self.alpha*Ica - self.Kca*Ca)
        #update Y
        self.derivS()
        self.Y+=self.dY*dt
        

class temporal_mdl(Red3,Red6):
    def __init__(self,noise=0.0,mpi=False,model='Red3'):
        """Model init."""
        #Generic elements
        self.mpi = mpi
        self.model = model
        if mpi:
            self.mpi_init()
        
    def mpi_init(self):
        #find the engine processes
        rc = Client(profile='mpi')
        rc.clear()
        #Create a view of the processes
        self.view = rc[:]

        #number of clients
        nCl = len(rc.ids)
        #divisors of nCl
        div = [i for i in range(1,nCl+1) if nCl%i==0]
        ldiv = len(div)

        #the surface will be divided into nbx rows and nby columns
        if ldiv %2 == 0:
            self.nbx = div[ldiv/2]
            self.nby = div[ldiv/2-1]
        else:
            self.nbx = self.nby = div[ldiv/2]



    def compute(self,tmax,Nx,Ny=0,Nz=0,stimCoord=[0,0,0,0],stimCoord2=[0,0,0,0]):
        if self.mpi:
            self.compute_mpi(tmax,Nx,Ny,Nz,stimCoord,stimCoord2)
        else:
            self.compute_serial(tmax,Nx,Ny,Nz,stimCoord,stimCoord2)

    def compute_serial(self,tmax,Nx,Ny,Nz,stimCoord,stimCoord2):
        if self.model == 'Red3':
           Red3.__init__(self,Nx,Ny,Nz) 
        elif self.model == 'Red6':
           Red6.__init__(self,Nx,Ny,Nz)

        self.stimCoord = stimCoord
        self.stimCoord = stimCoord2

        decim=10
        NbIter=0
        dt=0.05
        Ft = 0.15
        dtMin = dt
        dtMax = 6
        dVmax = 1

        #Initialise storage variables
        self.t=numpy.zeros(round(tmax/(dt*decim))+1)
        self.Vm=numpy.zeros((Nx+4,Ny+4,Nz+4,round(tmax/(dt*decim))+1))
        #Integration
        while self.time<tmax:
            Ist=0.2/2*(numpy.sign(numpy.sin(2*numpy.pi*self.time/(1*tmax)))+1)*numpy.sin(2*numpy.pi*self.time/(1*tmax))
            self.Istim[self.stimCoord]=Ist
           # mdl.Istim[50:95,100]=Ist
            self.derivT(dt)
            #define new time step
            dt = dtMin*dVmax/numpy.max(abs(self.dY[...,0].all())-Ft);
            if dt > dtMax:
                dt = dtMax
            if dt < dtMin:
                dt = dtMin
            self.time+=dt
            #stores time and state 
            if not round(self.time/dt)%decim:
                NbIter+=1
                self.t[NbIter]=self.time
                self.Vm[...,NbIter]=self.Y[...,0].copy()

    def compute_mpi(self,tmax,Nx,Ny,Nz,stimCoord,stimCoord2):

        def parallelcomp(tmax,Nx,Ny,Nz,nbx,nby,stimCoord,stimCoord2,model):
            """Main function lauched by the engine processes. Compute the euler integration
            of a block of the global surface

            Syntax:
            Results = parallelcomp(tmax,nbx,nby)
            
            Inputs:
            tmax -- duration of the simulation
            nbx -- number of rows
            nby -- number of columns

            Outputs:
            Results -- list of the results for each process
                    An element of this list is a dictionnary containing:
                        'Nx': Number of rows computed by this process
                        'Ny': Number of columns
                        'rank': rank of the process
                        'time': list of the time points
                        'x': indices of the first and last rows computed by this process,
                                according the global surface
                        'y': indices of the first and last columns
                        'Vm': solution
            """
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

#            def comm(to_send1,to_send2,bools):
#                from mpi4py import MPI
#                pair,first,last = bools
#                if pair:
#                    if first:
#                        MPI.COMM_WORLD.send(to_send_1 , dest=rank-1)
#                    if last:
#                        pass
#                        to_recv_1 = MPI.COMM_WORLD.recv(source=rank+1)
#                        MPI.COMM_WORLD.send(to_send_2, dest=rank+1)
#                    if first:
#                        pass
#                        to_recv_2 = MPI.COMM_WORLD.recv(source=rank-1)
#                else:
#                    if last:
#                        to_recv_1 = MPI.COMM_WORLD.recv(source=rank+1)
#                    if first:
#                        MPI.COMM_WORLD.send(to_send_1 , dest=rank-1)
#                        to_recv_2 = MPI.COMM_WORLD.recv(source=rank-1)
#                    if last:
#                        MPI.COMM_WORLD.send(to_send_2, dest=rank+1)
#                return to_recv1,to_recv2

            #dimensions of the surgace
            # Nx should be also greater than Ny
#                flag_swap = False
#                if Nx < Ny:
#                    Nx,Ny = Ny,Nx
#                    flag_swap = True


            rank = MPI.COMM_WORLD.Get_rank()
            # Which rows should I compute?
            [x,newNx] = findlimitsx(rank,nbx,Nx+4)

            # What about the columns?
            [y,newNy] = findlimitsy(rank,nby,Ny+4,nbx)


            #Stimulation (global coordinates)
            xyIstim1 = [3,42,4,6] 
            xyIstim2 = [40,82,95,97]
            
            #computing the local coordinates
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


            #Creation of the model (one for each process)
            mpi=[(rank%nbx==0),rank%nbx==(nbx-1),(rank/nbx==0),(rank/nbx==(nby-1)),True,True]
            if model == 'Red6':
                mdl=cell_mdl.Red6(Nx=newNx-2*(rank%nbx==0)-2*(rank%nbx==(nbx-1)),Ny=newNy-2*(rank/nbx==0)-2*(rank/nbx==(nby-1)),Nz=Nz,borders=mpi)
            elif model == 'Red3':
                mdl=cell_mdl.Red3(Nx=newNx-2*(rank%nbx==0)-2*(rank%nbx==(nbx-1)),Ny=newNy-2*(rank/nbx==0)-2*(rank/nbx==(nby-1)),Nz=Nz,borders=mpi)

            #Tells the model where the stimuli are
            if xyIstim1[0] != -1 and xyIstim1[2] != -1:
                mdl.stimCoord = xyIstim1
            if xyIstim2[0] != -1 and xyIstim2[2] != -1:
                mdl.stimCoord2 = xyIstim2


            decim=20
            NbIter=0
            dt=0.05
            Ft = 0.15

            time=numpy.zeros(round(tmax/(dt*decim))+1)

            if Nx*Ny*Nz:
                Vm=numpy.zeros((mdl.Nx,mdl.Ny,mdl.Nz,round(tmax/(dt*decim))+1))
            elif Nx*Ny:
                Vm=numpy.zeros((mdl.Nx,mdl.Ny,round(tmax/(dt*decim))+1))
            elif Nx:
                Vm=numpy.zeros((mdl.Nx,round(tmax/(dt*decim))+1))

            while (mdl.time<tmax):
                Ist=0.2/2*(numpy.sign(numpy.sin(2*numpy.pi*mdl.time/(1*tmax)))+1)*numpy.sin(2*numpy.pi*mdl.time/(1*tmax))

                if xyIstim1[0] != -1 and xyIstim1[2] != -1:
                    mdl.Istim[xyIstim1[0]:xyIstim1[1],xyIstim1[2]:xyIstim1[3]]=Ist
                if xyIstim2[0] != -1 and xyIstim2[2] != -1:
                    mdl.Istim[xyIstim2[0]:xyIstim2[1],xyIstim2[2]:xyIstim2[3]]=Ist

                mdl.derivT(dt)

                #Communication 
                if Nx*Ny*Nz:
                    to_send_1 = mdl.Y[0,:,:,0]
                    to_send_2 = mdl.Y[-1,:,:,0]
                    to_recv_1 = mdl.Y[-1,:,:,0]
                    to_recv_2 = mdl.Y[0,:,:,0]

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

                    mdl.Y[-1,:,:,0] = to_recv_1 
                    mdl.Y[0,:,:,0] =  to_recv_2

                    #Communication y
                    to_send_1 = mdl.Y[:,1,:,0]
                    to_send_2 = mdl.Y[:,-2,:,0]
                    to_recv_1 = mdl.Y[:,-1,:,0]
                    to_recv_2 = mdl.Y[:,0,:,0]

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

                    mdl.Y[:,-1,:,0] = to_recv_1
                    mdl.Y[:,0,:,0] =  to_recv_2




                elif Nx*Ny:
                    to_send_1 = mdl.Y[0,:,0]
                    to_send_2 = mdl.Y[-1,:,0]
                    to_recv_1 = mdl.Y[-1,:,0]
                    to_recv_2 = mdl.Y[0,:,0]

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

                    mdl.Y[-1,:,0] = to_recv_1 
                    mdl.Y[0,:,0] =  to_recv_2

                    #Communication y
                    to_send_1 = mdl.Y[:,1,0]
                    to_send_2 = mdl.Y[:,-2,0]
                    to_recv_1 = mdl.Y[:,-1,0]
                    to_recv_2 = mdl.Y[:,0,0]

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

                    mdl.Y[:,-1,0] = to_recv_1
                    mdl.Y[:,0,0] =  to_recv_2



                elif Nx:
                    to_send1,to_send2 = mdl.Y[0,0],mdl.Y[-1,0]
                    pair,first,last = [rank%2,rank%nbx!=0,rank%nbx!= nbx-1]
                    if pair:
                        if first:
                            MPI.COMM_WORLD.send(to_send_1 , dest=rank-1)
                        if last:
                            to_recv_1 = MPI.COMM_WORLD.recv(source=rank+1)
                            MPI.COMM_WORLD.send(to_send_2, dest=rank+1)
                        if first:
                            to_recv_2 = MPI.COMM_WORLD.recv(source=rank-1)
                    else:
                        if last:
                            to_recv_1 = MPI.COMM_WORLD.recv(source=rank+1)
                        if first:
                            MPI.COMM_WORLD.send(to_send_1 , dest=rank-1)
                            to_recv_2 = MPI.COMM_WORLD.recv(source=rank-1)
                        if last:
                            MPI.COMM_WORLD.send(to_send_2, dest=rank+1)
                    mdl.Y[-1,0],mdl.Y[0,0] = to_recv1,to_recv2



                mdl.time =mdl.time+dt
                if not round(mdl.time/dt)%decim:
                    NbIter+=1
                    time[NbIter]=mdl.time
                    if Nx*Ny*Nz:
                        Vm[:,:,:,NbIter]=mdl.Y[:,:,:,0].copy()
                    elif Nx*Ny:
                        Vm[:,:,NbIter]=mdl.Y[:,:,0].copy()
                    elif Nx:
                        Vm[:,NbIter]=mdl.Y[:,0].copy()


            return {'rank':rank,'time':time,'x':x,'y':y,'Vm':Vm}




        res = self.view.apply_async(parallelcomp,tmax,Nx,Ny,Nz,self.nbx,self.nby,stimCoord,stimCoord2,self.model)
        self.view.wait(res)  #wait for the results
        tabResults = res.get()

        self.t = tabResults[0]['time']
        tabrank = numpy.empty(len(tabResults))
        for i in range(len(tabResults)):
            tabrank[i] = tabResults[i]['rank']

        def find(f, seq):
            comp = 0
            for item in seq:
	            if item == f: 
		            return comp
	            comp += 1
            return -1

        # Aggregation of the results
        if Nx*Ny*Nz:
            self.Vm = numpy.empty((Nx+4,Ny+4,Nz+4,len(self.t)))
        elif Nx*Ny:
            self.Vm = numpy.empty((Nx+4,Ny+4,len(self.t)))
        elif Nx:
            self.Vm = numpy.empty((Nx+4,len(self.t)))

        for i in range(len(tabrank)):
            i_client = find(i, tabrank)
            x = tabResults[i_client]['x']
            y = tabResults[i_client]['y']
            self.Vm[x[0]:x[1],y[0]:y[1],...]=numpy.array(tabResults[i_client]['Vm'])

