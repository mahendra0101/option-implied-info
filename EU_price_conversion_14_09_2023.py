#%% 
"""
@author: Mahendra Kumar Singh (mahenks1@iastate.edu) 
@Date: 14SEP2023   
"""  
#%% 
#Clear all the variables from the workspace. 
from IPython import get_ipython
get_ipython().magic('reset -sf') 
import numpy as np
import pandas as pd
import scipy.stats as si   
import os                
#%%
#Objective:- Convert American Futures Option Prices to their European futures option prices using Chaudhury (1995) MA method. 
#%%
#Basic data formatting
print(os.getcwd())  
#Locate the directory where all the grand datasets are located. 
os.chdir("M:\\MAHENKS1\\study_stuff\\Commodity_Markets\\PhD_Thesis_Work\\Chapter_1_Volatility\\work_in_progress\\\data_analysis\\2023_09_18\\vix_svix_suitable")  
#Change the input file name over here.    
#mydata=pd.read_stata('grand_corn_american_18_09_2023.dta')  
#mydata=pd.read_stata('grand_soybean_american_18_09_2023.dta')
#mydata=pd.read_stata('grand_wheat_american_18_09_2023.dta')
mydata=pd.read_stata('grand_crude_oil_american_18_09_2023.dta')   
print(os.getcwd())  
print(mydata.head())
print(mydata.tail())
#print(mydata)
print(type(mydata))  #print the data type imported. 
print(list(mydata))  #print all the variable names in the dataframe
print(type(mydata.date))
list_date=np.unique(mydata.date1)   #list of unique dates in the data
print(list_date)
print(type(list_date))      
#%% 
#Bisection method based solution for Chaudhury (1995) MA technique to convert American Futures Option prices to European Futures Options prices. 
#Moreover, it implicitly computes the Black (1976) based implied volatility from the converted european prices.  

def bisection_call(F, K, T, l, r):
    
    #F: futures price
    #K: strike price
    #T: time to maturity  (as per the VIX white paper guidelines)
    #l: American Call Futures Option price 
    #r: interest rate
    
    a=0.0000000001 
    b=5
    sigma=0.5 
    
    def func(sigma):
        d1 = (np.log(F/K) + (0.5*sigma**2)*T)/(sigma*np.sqrt(T))
        d2 = d1-(sigma*np.sqrt(T)) 
        return (np.exp(-r*T)*(F*si.norm.cdf(d1, 0.0, 1.0) - K*si.norm.cdf(d2, 0.0, 1.0)))*(np.exp(0.5*si.norm.cdf(d2, 0.0, 1.0)*r*T)) - l  
  
    if (func(a)*func(b)>0):
        print("You have not assumed right a and b\n")
        return np.nan, np.nan 
  
    while ((b-a)>=0.0000001):
 
        # Find middle point
        sigma=(a+b)/2

        # Check if middle point is root
        if (abs(func(sigma))<=0.00001):
            break
  
        #Decide the sub-interval for the next iteration
        if (func(sigma)*func(a)<0):
            b=sigma
        else:
            a=sigma 
        
    return func(sigma)+l, sigma 
    
def bisection_put(F, K, T, l, r):
    
    #F: futures price
    #K: strike price
    #T: time to maturity  (as per the VIX white paper guidelines)
    #l: American Put Futures Option value
    #r: interest rate
    
    a=0.0000000001 
    b=5
    sigma=0.5 
    
    def func(sigma):
        d1 = (np.log(F/K) + (0.5*sigma**2)*T)/(sigma*np.sqrt(T))
        d2 = d1-(sigma*np.sqrt(T)) 
        return (np.exp(-r*T)*(K*si.norm.cdf(-d2, 0.0, 1.0) - F*si.norm.cdf(-d1, 0.0, 1.0)))*(np.exp(0.5*si.norm.cdf(-d2, 0.0, 1.0)*r*T)) - l  
  
    if (func(a)*func(b)>0):
        print("You have not assumed right a and b\n")
        return np.nan, np.nan 
  
    while ((b-a)>=0.0000001):
 
        # Find middle point
        sigma=(a+b)/2

        # Check if middle point is root
        if (abs(func(sigma))<=0.00001):
            break
  
        #Decide the sub-interval for the next iteration
        if (func(sigma)*func(a)<0):
            b=sigma
        else:
            a=sigma 
        
    return func(sigma)+l, sigma 
        
    
#%% 
#Main computation begins!    

#Drop observations with zero strikes. They are simply clerical errors and create issues with BSIV estimation. 
filtered_df=mydata.loc[(mydata["strike"]>0)]       
#calculate the time to expiration
filtered_df['date']= pd.to_datetime(filtered_df['date'])  
filtered_df["time"]=(filtered_df["option_exp"]-filtered_df["date"]) 
filtered_df["days"]= filtered_df["time"].dt.days
#calculate the time to expiration in minutes per year
filtered_df["minute"]=filtered_df["days"]*(24*60/525600)

# Compute BSIV and European price for each option available in the data
for index, row in filtered_df.iterrows(): 
    F=row['fut_price']
    K=row['strike']
    T=row['minute']
    l=row['price']
    r=row['interest_rate']

    #Now call the bisection method algorithm that could compute european futures option prices using MA method. 
    #Call futures option. 
    if row['type']=="C":
        eu_price, bsiv=bisection_call(F, K, T, l, r) 
        filtered_df.loc[index,'eu_price']=eu_price
        filtered_df.loc[index,'bsiv']=bsiv
    #Put futures option. 
    if row['type']=="P":
        eu_price, bsiv=bisection_put(F, K, T, l, r) 
        filtered_df.loc[index,'eu_price']=eu_price
        filtered_df.loc[index,'bsiv']=bsiv
   
        
#%% 
#Change working directory before exporting the data. 
os.getcwd()
os.chdir("M:\\MAHENKS1\\study_stuff\\Commodity_Markets\\PhD_Thesis_Work\\Chapter_1_Volatility\\work_in_progress\\data_analysis\\2023_09_19") 
os.getcwd()  

#export final data

#file_name="corn_converted_price.csv"
#file_name="soybean_converted_price.csv"
#file_name="wheat_converted_price.csv"
file_name="crude_oil_converted_price.csv"
#create the Excel file. 
filtered_df.to_csv(file_name, index=False)
                                    
            
#%% The End #%% 



