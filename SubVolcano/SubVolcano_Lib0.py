# Author Martín Manuel Gómez Míguez
# mail mamagomi@gmail.com
# GitHub: @Correlo

#Packages from Python
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
import numpy.fft as FFT
import glob as glob

#Param file
from Param import *

def TIME(data, t_c=c_d_t_tuple):
    '''
    Function to obtain the time of the measurements in s 

    Input    

    data        -> data from data set (ndarray)

    Optional Input

    t_c         -> indexes of columns with time (tuple)

    Output

    Data        -> array with data from the file (ndarray)
    '''

    return (data[:,t_c[0]]*3600+data[:,t_c[1]]*60+data[:,t_c[2]]).T


def digitalhour(Data, t_step, t_c=c_d_t_tuple):
    '''
    Function to create a label with digital time

    Input    

    Data        -> array with data from the file (ndarray)
    t_step      -> time step in minutes to time ticks (int)

    Optional Input 

    t_c         -> indexes of columns with time (tuple)

    Output

    digital     -> list of the label of the ticks (list)
    ti          -> time lower limit in s (float)
    tf          -> time upper limit in s (float)
    '''

    #Data from data
    hi=int(Data[0,0])    #First measurement hour
    mini=int(Data[0,1])  #First measurement minute
    hf=int(Data[-1,0])   #Last measurement hour
    minf=int(Data[-1,1]) #Last measurement minute
    
    #Avoid problems with initial files
    if hi==23:
        hi=0;mini=0;

    minf+=(hf-hi)*60 #Hour correction


    #Round lower limit using ceil() and upper limit with floor 
    #according to t_step criteria

    lower=np.ceil(mini/t_step)*t_step
    upper=np.floor(minf/t_step)*t_step

    #Limits of xlim
    ti=hi*3600+lower*60
    tf=hf*3600+upper*60

    #Obtain minute array
    min_array=np.arange(lower,upper + t_step, t_step, dtype=int)
    
    #Build string list with hours:
    digital=[]
    for min in min_array:
        #Correct hours and minutes
        h=hi+int(min/60)
        min-=int(min/60)*60

        hour=''
        hour+=str(h)
    
        if h<10:
            hour= '0' + hour
        hour+=':'
           
        if min<10:
            hour+='0'+str(min)
        else:
            hour+=str(min)
        
        digital.append(hour)

    return [digital,ti,tf]


def Difference(A,n):
    '''
    Function to obtain the difference between two values in the array 
    separated by a distance n

    Input    

    A           -> (array)
    n           -> index lag (int)

    Output

    Diff        -> difference between selected elements (array)
    '''
    
    B=np.roll(A,-n)

    return (B-A)[:-n]

def W(D,X):
    '''
    Function to measure the variance

    Input    

    D           -> index of lag time (array)
    X           -> sample of measurements (list of ndarray)

    Output

    W1-W2       -> Variance for a certain lag time (float)
    '''
    #Create auxiliar array
    aW1=np.zeros(len(X))
    aW2=np.zeros(len(X))
    
    #Loop over each sample of measurements in the list
    for i in range(len(X)):
        Diff=Difference(X[i],D)
        aW1[i]=np.mean(Diff**2)
        aW2[i]=np.mean(Diff)
    
    #Terms of the variance 
    W1=np.mean(aW1)
    W2=np.mean(aW2)**2

    return W1-W2 
 

def Power_Delta(Delta, Var_Mag, bounds, cov=False): 
    '''
    Function to perform a linear fit for the variance

    Input    

    Delta       -> Lag time in s (array)
    Var_mag     -> Variance of a magnitude (array)
    bounds      -> Bounds of the region (tuple like (b_inf,b_sup))

    Optional Input 
    cov         -> Obtain covariance matrix (bool)

    Output

    p           -> Coefficients of the fit (array like [(p1,p0)])
    x           -> x-data in log scale (array)
    Cov         -> Covariance matrix (ndarray)
    '''

    #Time limits of each set
    inf=bounds[0]
    sup=bounds[1]

    #Select data
    tinf=Delta>inf
    tsup=Delta<sup

    x=Delta[tinf*tsup]
    y=Var_Mag[tinf*tsup]
       
    #Fit
    if cov:
        p,Cov=np.polyfit(x,y,1,cov=cov)
    else:
        p=np.polyfit(x,y,1,cov=cov)

    
    if cov:
        return [p,x,Cov]
    else:
        return [p,x]


def Fourier(f_t,t,t_step):
    '''
    Function to perform the FFT 
    
    Input    

    f_t         -> function of time (array)
    t           -> time in s (array)
    t_step      -> time step of the machine in s (float)

    Output

    Result      -> c0 -> frq; c1 -> PS;
    '''

    #Perform FFT
    delta_v=FFT.fft(f_t)/len(f_t)
    #Shift measurements
    delta_v=FFT.fftshift(delta_v)
    #Obtain the Power Spectrum
    Pv=np.real(delta_v*delta_v.conjugate())
    #Obtain and shift frequencies
    v=FFT.fftshift(FFT.fftfreq(len(t),t_step))

    #Take only the positive frequencies
    Pv_p=Pv[v>=0]
    v_p=v[v>=0]

    Result=[v_p,Pv_p]
    
    return Result


#Function to perform a DFT or a NDFT measurement 
def NDFT(f_t , t):
    '''
    Function to perform the NDFT
    
    Input    
    f_t         -> function of time (array)
    t           -> time in s (array)

    Output

    Result      -> c0 -> frq; c1 -> f_v;
    '''
    
    #Adimensional frequencies
    N=len(t)
    n=t/t[-1]*N
    M=n-n[-1]/2
    M=M[M>0]
    #Dimensional
    freq=M/t[-1]
    
    #Initialize the array of Fourier transform
    f_v=np.array([])
    
    #Loop over each Fourier mode
    for m in M:
        arg=2*np.pi*n*m/N
        Exp=np.e**(-1j*arg)
        fv=np.mean(f_t*Exp)
        f_v=np.append(f_v,fv)  

    Result=[freq, f_v] 

    return Result










