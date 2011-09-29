"""Tissue models classes:
Tissuemodel: Generic base class, not a functionnal model by itself.
Red3: Uses reduced 3 vars uterine cell model (J.Laforet).
Red6: Uses reduced 6 vars uterine cell model (S.Rihana)."""

import numpy
import pylab
from scipy.ndimage.filters import correlate1d


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
        
