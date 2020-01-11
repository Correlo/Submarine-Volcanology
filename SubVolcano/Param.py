# Author Martín Manuel Gómez Míguez
# mail mamagomi@gmail.com

#-+- coding: utf -+-

'''Data parameters'''

#Columns of the file
c_f_h=4      #Hours            
c_f_min=5    #Miniutes         
c_f_s=6      #Seconds          
c_f_T=8      #Temperature      
c_f_C=9      #Conductivity
c_f_D=10     #Depth
c_f_S=11     #Salinity
c_f_SS=12    #Speed of Sound

#Tuple with the columns of the file
c_tuple=(c_f_h,c_f_min,c_f_s,c_f_T,c_f_C,c_f_D,c_f_S,c_f_SS)

#Delimiters
delimiters=[':',',','/']

#Columns of the data array from the data set
c_d_h=0      #Hours     
c_d_min=1    #Minutes
c_d_s=2      #Seconds
c_d_T=3      #Temperature
c_d_C=4      #Conductivity
c_d_D=5      #Depth
c_d_S=6      #Salinity
c_d_SS=7     #Speed of Sound

#Tuple with the columns of time in the array
c_d_t_tuple=(c_d_h,c_d_min,c_d_s)
#Tuple with the columns of magnitudes in the array
c_d_mag_tuple=(c_d_T,c_d_C,c_d_S,c_d_SS)

#labels of axis
maglabel=['Temperature (ºC)','Conductivity (S/m)','Salinity (PSU)','Speed of sound (m/s)']
sqlabel=[r'$ºC^2$',r'$(S/m)^2$',r'$PSU^2$',r'$(m/s)^2$']


