# Author Martín Manuel Gómez Míguez
# mail mamagomi@gmail.com
#GitHub: @Correlo

#Packages from Python
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
import numpy.fft as FFT
import glob as glob
import scipy.stats as st
from matplotlib.ticker import MaxNLocator

#Packages from Martín
from SubVolcano_Lib0 import * 

#Function to read a datafile

def Read(filename, columns=c_tuple, delimiters=delimiters):
    '''
    Function to read the data set

    Input    

    filename    -> name of the file (string)
    
    Optional Input

    columns     -> index of columns of the file (tuple)
    delimiters  -> characters to delimit columns that are not spaces (list)

    Output

    Data        -> array with data from the file (ndarray)
    '''

    #Open data file
    f=open(filename,'r')

    #Read data
    s = f.read()

    #Change delimiters for spaces
    for a in delimiters:
        s = s.replace(a,' ')

    #Take the columns that we need
    Data=np.loadtxt(StringIO(s),usecols=columns)

    #Close file
    f.close()

    return Data


def FolderRead(foldername, columns=c_tuple, delimiters=delimiters):
    '''
    Function to read the data set contained in a folder
    and concatenate the data

    Input    

    foldername  -> name of the folder + '/' (string)

    Optional Input

    columns     -> index of columns of the file (tuple)
    delimiters  -> characters to delimit columns that are not spaces (list)

    Output

    Data        -> array with data from the file (ndarray)
    '''

    #Obtain the name of the data files and sort them
    files=glob.glob(foldername + '*')
    files.sort()

    #Obtain the dataset and concatenate all data in one array
    Data=Read(files[0])

    for name in files[1:-1]:
        Dataread=Read(name, columns=c_tuple, delimiters=delimiters)
        Data=np.vstack((Data,Dataread))

    return Data


def Dvst(Data, t_step, Plotfolder, label, t_c=c_d_t_tuple, d_c=c_d_D):
    '''
    Function to plot depth vs time

    Input    

    Data        -> array with data from the file (ndarray)
    t_step      -> time step in minutes to time ticks (int)
    Plotfolder  -> name of the folder to save the figure (string)
    label       -> label to distinguish the figure (string)

    Optional Input

    t_c         -> indexes of columns with time (tuple)
    d_c         -> index of column with depth (int)

    Output

    Only execute the action
    '''

    #Obtain time array
    Tinsec=TIME(Data, t_c)

    #Obtain the axis of the plot
    digital,ti,tf=digitalhour(Data, t_step, t_c)

    xlim=np.arange(ti,tf+t_step*60,t_step*60)

    #Depth vs Time 
    plt.figure(1,figsize=(10,5))
    plt.plot(Tinsec,Data[:,d_c],'.',markersize=1)
    plt.gca().invert_yaxis()
    plt.xlabel('Time', fontsize=15)
    plt.ylabel('Depth (m)', fontsize=15)
    plt.tick_params(axis='both',direction='inout',which='minor',length=5,width=1,labelsize=14)
    plt.tick_params(axis='both',direction='inout',which='major',length=8,width=1,labelsize=14)
    plt.minorticks_on()
    plt.xticks(xlim, digital)
    plt.xlim(Tinsec[0],Tinsec[-1])
    plt.tight_layout()

    #Save figure and close it
    plt.savefig(Plotfolder + '/' +label + 'Dvst.jpg')
    plt.close()


def Profiles(Data, Plotfolder, label, t_c=c_d_t_tuple, d_c=c_d_D, mag_c=c_d_mag_tuple,
             Xlabel=maglabel):
    '''
    Function to plot the profiles of each magnitude

    Input    

    Data        -> array with data from the file (ndarray)
    Plotfolder  -> name of the folder to save the figure (string)
    label       -> label to distinguish the figure (string)

    Optional Input

    t_c         -> indexes of columns with time (tuple)
    d_c         -> index of column with depth (int)
    mag_c       -> index of columns with magnitudes (tuple)
    Xlabel      -> labels of x-axis (list)

    Output

    Only execute the action
    '''

    #Obtain time array
    Tinsec=TIME(Data, t_c)
    Tinsec-=Tinsec[0]
 
    for i in range(len(Xlabel)):

        #Plot
        fig=plt.figure(figsize=(5,5))
        ax=fig.add_subplot(1,1,1)
        plt.plot(Data[:,mag_c[i]],Data[:,d_c],'.',markersize=1)
        plt.gca().invert_yaxis()
        plt.xlabel(Xlabel[i], fontsize=15)
        plt.ylabel('Depth (m)', fontsize=15)
        plt.minorticks_on()
        ax.xaxis.set_major_locator(MaxNLocator(6))
        ax.xaxis.set_minor_locator(MaxNLocator(30))
        ax.tick_params(axis='both',direction='inout',which='minor',length=5,width=1,labelsize=14)
        ax.tick_params(axis='both',direction='inout',which='major',length=8,width=1,labelsize=14)
        plt.tight_layout()
        plt.savefig(Plotfolder + '/' + label + 'Profile' + str(i) + '.jpg')
        plt.close()


def Cutter(Data, time_range, t_c=c_d_t_tuple):
    '''
    Function to select data in a certain range of time
    Input    

    Data        -> array with data from the file (ndarray)
    time_range  -> lower and upper time limits (tuple like ((h0,min0),(h1,min1)))

    Optional Input

    t_c         -> indexes of columns with time (tuple)

    Output

    Cutdata     -> array with data from the file (ndarray)
    '''

    #Obtain bounds
    h0=time_range[0][0]
    min0=time_range[0][1]
    h1=time_range[1][0]
    min1=time_range[1][1]

    #Obtain time array
    Tinsec=TIME(Data, t_c)

    #Obtain time limits
    t0 = h0*3600 + min0*60
    t1 = h1*3600 + min1*60

    t_inf=Tinsec>t0
    t_sup=Tinsec<t1

    #Take the data that you want
    Cutdata=Data[t_inf*t_sup,:]

    return Cutdata


def time_series(Cutfile, Plotfolder, t_step, t_c=c_d_t_tuple, mag_c=c_d_mag_tuple, Ylabel=maglabel):
    '''
    Function to plot the time series
    Input    

    Cutfile     -> name of the file (string)
    Plotfolder  -> name of the folder to save the figure (string)
    t_step      -> time step in minutes to time ticks (int)

    Optional Input

    t_c         -> indexes of columns with time (tuple)
    mag_c       -> index of columns with magnitudes (tuple)
    Ylabel      -> labels of x-axis (list)

    Output

    Only execute the action
    '''

    #Take data from Cutfile
    Cutdata=np.loadtxt(Cutfile)
    time=TIME(Cutdata, t_c)
    
    #Obtain the axis of the plot
    digital,ti,tf=digitalhour(Cutdata, t_step, t_c)

    Xlim=np.arange(ti,tf+t_step*60,t_step*60)

    #Loop over magnitudes
    for i in range(len(Ylabel)):

        #Plot
        plt.figure(figsize=(6,5))
        plt.plot(time, Cutdata[:,mag_c[i]],'.',markersize=1)
        plt.xlabel('Time', fontsize=15)
        plt.ylabel(Ylabel[i], fontsize=15)
        plt.xticks(Xlim, digital)
        plt.xlim(time[0],time[-1])
        plt.minorticks_on()
        plt.tick_params(axis='both',direction='inout',which='minor',length=5,width=1,labelsize=14)
        plt.tick_params(axis='both',direction='inout',which='major',length=8,width=1,labelsize=14)
        plt.tight_layout()
        plt.savefig(Plotfolder + 'TS_' + str(i) + '.jpg')
        plt.close()


def Var(Cutfile, delta_max, t_c=c_d_t_tuple, mag_c=c_d_mag_tuple):
    '''
    Function to measure W(Delta) for each physical magnitude

    Input    

    Cutfile     -> name of the file (string)
    delta_max   -> upper limit of the index of lag time (int)

    Optional Input

    t_c         -> indexes of columns with time (tuple)
    mag_c       -> index of columns with magnitudes (tuple)

    Output

    Result      -> Variance of each magnitude and lag time (ndarray)
                   c0 -> lag time; c1 -> var. mag. 0; c2 -> ...
    '''

    #Import Cutfile data
    data=np.loadtxt(Cutfile)
    
    #Get the time step
    tau,_=st.mode(Difference(data[:,t_c[2]],1))
    tau=tau[0]

    #Create Delta array
    DD=np.arange(1,delta_max+1,1,dtype=int)

    #Prepare output
    Result=[DD*tau]

    #Loop over Magnitudes
    for i in range(len(mag_c)):
        
        W_res=np.array([])
        #Loop over Delta array
        for D in DD:
            W_res=np.append(W_res,W(D,[data[:,mag_c[i]]]))
        
        Result.append(W_res)

    Result=np.array(Result).T

    return Result


def Varplot(Cutfile, delta_max, Plotfolder, t_c=c_d_t_tuple, mag_c=c_d_mag_tuple, var_l=sqlabel):
    '''
    Function to plot W(Delta) for each physical magnitude

    Input    

    Cutfile     -> name of the file (string)
    delta_max   -> upper limit of the index of lag time (int)
    Plotfolder  -> name of the folder to save the figure (string)

    Optional Input

    t_c         -> indexes of columns with time (tuple)
    mag_c       -> index of columns with magnitudes (tuple)
    var_l       -> dimension of the variance measurement (list)

    Output

    Only execute the action
    '''

    #Import Cutfile data
    data=np.loadtxt(Cutfile)
    
    #Get the time step
    tau,_=st.mode(Difference(data[:,t_c[2]],1))
    tau=tau[0]

    #Create Delta array
    DD=np.arange(1,delta_max,1,dtype=int)

    #Loop over Magnitudes
    for i in range(len(mag_c)):
        
        W_res=np.array([])
        #Loop over Delta array
        for D in DD:
            W_res=np.append(W_res,W(D,[data[:,mag_c[i]]]))

        plt.plot(DD*tau,W_res,'.')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$\Delta$ $(s)$', fontsize=15)
        plt.xlim(min(DD*tau)-0.05*min(DD*tau),max(DD*tau)+0.05*max(DD*tau))
        plt.ylabel(r'$W(\Delta)$' + ' (' + var_l[i] + ')', fontsize=15)
        plt.tick_params(axis='both',direction='inout',which='minor',length=5,width=1,labelsize=14)
        plt.tick_params(axis='both',direction='inout',which='major',length=8,width=1,labelsize=14)
        plt.tight_layout()
        plt.savefig(Plotfolder + 'W' + str(i) + '.jpg')
        plt.close()
        
              
def Power_Delta_plot(Delta, Var_Mag, Sets, Plotfile, Units):
    '''
    Function to perform and plot a linear fit for the variance

    Input    

    Delta       -> Lag time in s (array)
    Var_mag     -> Variance of a magnitude (array)
    Sets        -> List of bounds of the regions to perform a fit 
                   (tuple like [(b0_inf,b0_sup), (b1_inf,b1_sup), ...])
    Plotfile    -> Name of the file of the figure (str)
    Units       -> Units of the variance (str)

    Output

    Only execute the action
    '''

    #Log10 of Var_Mag
    DLog10=np.log10(Delta)
    VLog10=np.log10(Var_Mag)

    plt.plot(Delta,Var_Mag,'.', markersize=3)

    P=np.array([])
    #Loop over each region
    for bounds in Sets:
        #Fit
        p,x=Power_Delta(DLog10, VLog10, np.log10(bounds))
        P=np.append(P,p)

        plt.plot(10**x, 10**p[1]*(10**x)**p[0],'-',markersize=20,label=r'$\Delta ^ {%.2f}$' %     
                 (p[0]))

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\Delta$ $(s)$', fontsize=15)
    plt.ylabel(r'$W(\Delta)$' + ' ('+ Units + ')', fontsize=15)
    plt.xlim(min(Delta)-0.05*min(Delta),max(Delta)+0.05*max(Delta))
    plt.legend(frameon=False)
    plt.tick_params(axis='both',direction='inout',which='minor',length=5,width=1,labelsize=14)
    plt.tick_params(axis='both',direction='inout',which='major',length=8,width=1,labelsize=14)
    plt.tight_layout()
    plt.savefig(Plotfile)
    plt.close()

    
def mom(Cutfile, delta_max, Q, mag_col, t_c=c_d_t_tuple):
    '''
    Function to measure the generalized moments

    Input    

    Cutfile     -> Name of the file (string)
    delta_max   -> Upper limit of the index of lag time (int)
    Q           -> Moments (array)
    mag_col     -> Column of the magnitud in Cutfile (int)

    Optional Input

    t_c         -> indexes of columns with time (tuple)

    Output

    Result      -> Lag time in s and the q-moments of the magnitude (ndarray)
                   c0 -> lag time; c1 -> mom Q[0]; ...
    '''

    #Import Cutfile data
    data=np.loadtxt(Cutfile)

    #Get the time step
    tau,_=st.mode(Difference(data[:,t_c[2]],1))
    tau=tau[0]

    #Create Delta array
    DD=np.arange(1,delta_max+1,1,dtype=int)
    
    #Organize data
    Mag=data[:,mag_col]
    
    #Output array
    Result=DD*tau
        
    mM=np.array([])
    #Loop over Delta array
    for D in DD:
            
        mMD=np.absolute(Difference(Mag,D))

        mMDa=np.array([])
        #Loop over Q array
        for q in Q:
            mMDq=np.mean(mMD**q)
            mMDa=np.append(mMDa,mMDq)
                
        mM=np.append(mM,mMDa)

        
    Result=np.hstack((Result.reshape(delta_max,1),mM.reshape((delta_max,len(Q)))))
              
    return Result
    

def plot_mom(Delta, Mom, bounds, Plotfolder, label):
    '''
    Function to plot the generalized moments
    
    Input    

    Delta       -> Lag time in s (array)
    Mom         -> Measured moments (ndarray with shape len(Delta) x len(Q))
    bounds      -> Bounds of the region (tuple like (b_inf,b_sup))
    Plotfolder  -> name of the folder to save the figure (string)
    label       -> label to distinguish the figure (string)

    Output

    Only execute the action
    '''
    #Plot
    plt.figure(figsize=(6,5))
    for momq in Mom.T:
        #Plot the moments
        plt.plot(Delta,momq,'.')
        #Perform a fit
        p,x=Power_Delta(np.log10(Delta), np.log10(momq), np.log10(bounds))
        plt.plot(10**x,10**p[1]*(10**x)**p[0])
        
      
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\Delta (s)$', fontsize=15)
    plt.ylabel(r'$\rho(q,\Delta)$', fontsize=15)
    plt.minorticks_on()
    plt.xlim(min(Delta)-0.05*min(Delta),max(Delta)+0.05*max(Delta))
    plt.tick_params(axis='both',direction='inout',which='minor',length=5,width=1,labelsize=14)
    plt.tick_params(axis='both',direction='inout',which='major',length=8,width=1,labelsize=14)
    plt.tight_layout()
    plt.savefig(Plotfolder + label + 'mom' + '.jpg')
    plt.close()


def struct(Delta, Mom, bounds, Q):
    '''
    Function to measure the structure function
    
    Input    

    Delta       -> Lag time in s (array)
    Mom         -> Measured moments (ndarray with shape len(Delta) x len(Q))
    bounds      -> Bounds of the region (tuple like (b_inf,b_sup))
    Q           -> Moments (array)

    Output
    
    Result      -> c0 -> q-moments; c1 -> z(q); c2 -> error (ndarray)
    ''' 
    #Measure the slope of the moments and the error
    Result=[]
    for momq in Mom.T:
        p,x,Cov=Power_Delta(np.log10(Delta), np.log10(momq),np.log10(bounds),cov=True)
        Result.append(np.array([p[0],np.sqrt(Cov[0,0])]))
    
    #Prepare the result
    Result=np.array(Result)
    Result=np.vstack((Q,Result.T))
    Result=Result.T

    return Result

def plot_struct(Q, Zq, Plotfile):
    '''
    Function to plot the structure function
    
    Input    

    Q           -> Moments (array)
    Zq          -> structure function (array)
    Plotfile    -> Name of the file of the figure (str)

    Output

    Only execute the action
    '''
    #Plot
    plt.figure(figsize=(6,5))
    plt.plot(Q, Zq,'.',markersize=5)
    plt.xlabel('q', fontsize=15)
    plt.ylabel('Z(q)', fontsize=15)
    plt.minorticks_on()
    plt.tick_params(axis='both',direction='inout',which='minor',length=5,width=1,labelsize=14)
    plt.tick_params(axis='both',direction='inout',which='major',length=8,width=1,labelsize=14)
    plt.tight_layout()
    plt.savefig(Plotfile)
    plt.close()
    

def FAnalysis(Data, t_step, t_c=c_d_t_tuple, mag_c=c_d_mag_tuple):
    '''
    Function to measure the Power Spectrum of the time series using FFT
    
    Input    

    Data        -> array with data from the file (ndarray)
    t_step      -> time step of the machine in s (float)

    Optional Input

    t_c         -> indexes of columns with time (tuple)
    mag_c       -> index of columns with magnitudes (tuple)

    Output

    Result      -> frequencies in Hz and normalized Power Spectrum (ndarray)
                   c0 -> frq of mag 0; c1 -> PS of mag 0;
                   c2 -> frq of mag 1; c3 -> PS of mag 1; ...
    '''
    #Obtain time array
    Tinsec=TIME(Data, t_c)
    t=Tinsec-Tinsec[0]

    Result=[]
    #Obtain the time series data
    for i in range(len(mag_c)):
        
        Mag=Data[:,mag_c[i]]
        delta_t=(Mag-np.mean(Mag))/np.mean(Mag)

        #Fourier analysis
        v_p,Pv_p=Fourier(delta_t,t,t_step)

        Result.append(v_p)
        Result.append(Pv_p)

    Result=np.array(Result).T

    return Result

def FAnalysis_plot(Data, t_step, Plotfolder, t_c=c_d_t_tuple, mag_c=c_d_mag_tuple):
    '''
    Function to plot the Power Spectrum of the time series using FFT
    
    Input    

    Data        -> array with data from the file (ndarray)
    t_step      -> time step of the machine in s (float)
    Plotfolder  -> name of the folder to save the figure (string)

    Optional Input

    t_c         -> indexes of columns with time (tuple)
    mag_c       -> index of columns with magnitudes (tuple)

    Output

    Only execute the action
    '''
    #Obtain time array
    Tinsec=TIME(Data,t_c)
    t=Tinsec-Tinsec[0]

    #Obtain the time serie data
    for i in range(len(mag_c)):
        
        Mag=Data[:,mag_c[i]]
        delta_t=(Mag-np.mean(Mag))/np.mean(Mag)

        #Fourier analysis
        v_p,Pv_p=Fourier(delta_t,t,t_step)

        #Plot in Fourier Space
        plt.figure(5,figsize=(6,5))
        plt.plot(v_p[1:], Pv_p[1:])
        plt.xlabel('Frequency (Hz)', fontsize=15)
        plt.ylabel(r'$|\delta(\nu)|^2$', fontsize=15)
        plt.yscale('log')
        plt.xlim(min(v_p),max(v_p))
        plt.minorticks_on()
        plt.tick_params(axis='both',direction='inout',which='minor',length=5,width=1,labelsize=14)
        plt.tick_params(axis='both',direction='inout',which='major',length=8,width=1,labelsize=14)
        plt.tight_layout()
        plt.savefig(Plotfolder + 'FFT_PS' + str(i) + '.jpg')
        plt.close()

def DFA(Data, t_c=c_d_t_tuple, mag_c=c_d_mag_tuple):
    '''
    Function to measure the Power Spectrum of the time series using NDFT
    
    Input    
    Data        -> array with data from the file (ndarray)

    Optional Input

    t_c         -> indexes of columns with time (tuple)
    mag_c       -> index of columns with magnitudes (tuple)

    Output

    Result      -> c0 -> frq; c1 -> f_v;
    '''
    #Obtain time array
    Tinsec=TIME(Data, t_c)
    t=Tinsec-Tinsec[0]

    Result=[]
    #Obtain the time serie data
    for i in range(len(mag_c)):
        
        Mag=Data[:,mag_c[i]]
        delta_t=(Mag-np.mean(Mag))/np.mean(Mag)

        #Fourier analysis
        v_p,delta_p=NDFT(delta_t, t)
        Pv_p=np.real(delta_p*delta_p.conjugate())

        Result.append(v_p)
        Result.append(Pv_p)

    Result=np.array(Result).T

    return Result


def plot_DFA(frq, PS, Plotfile):
    '''
    Function to plot the Power Spectrum of the time series using NDFT
    
    Input    
    frq         -> frequency in Hz (array)
    PS          -> Power Spectrum (array)

    Output

    Only execute the action
    '''
    #Plot in Fourier Space
    plt.figure(5,figsize=(6,5))
    plt.plot(frq[1:], PS[1:])
    plt.xlabel('Frequency (Hz)', fontsize=15)
    plt.ylabel(r'$|\delta(\nu)|^2$', fontsize=15)
    plt.yscale('log')
    plt.xlim(min(frq),max(frq))
    plt.minorticks_on()
    plt.tick_params(axis='both',direction='inout',which='minor',length=5,width=1,labelsize=14)
    plt.tick_params(axis='both',direction='inout',which='major',length=8,width=1,labelsize=14)
    plt.tight_layout()
    plt.savefig(Plotfile)
    plt.close()





