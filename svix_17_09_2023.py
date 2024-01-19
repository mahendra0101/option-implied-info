#%% 
"""
@author: Mahendra Kumar Singh (mahenks1@iastate.edu) 
@Date: 17 SEP 2023 
"""     
#%% 
#Clear all the variables from the workspace. 
from IPython import get_ipython
get_ipython().magic('reset -sf')   
import numpy as np
import pandas as pd
import scipy.stats as si
import math
import os       
#%%
#Note:- This code computes SVIX for the next provided time horizon on a given trading date.
# Just replaced strike price with underlying futures price in the lines 200-209.    
#%%
#Basic data formatting
print(os.getcwd())
#Locate the directory where all the grand datasets are located. 
os.chdir("M:\\MAHENKS1\\study_stuff\\Commodity_Markets\\PhD_Thesis_Work\\Chapter_1_Volatility\\work_in_progress\\data_analysis\\2023_09_19\\vix_svix_suitable")   
#Change the input data file name over here. 
#mydata=pd.read_stata('grand_corn_converted_19_09_2023.dta')  
#mydata=pd.read_stata('grand_soybean_converted_19_09_2023.dta')  
#mydata=pd.read_stata('grand_wheat_converted_19_09_2023.dta')  
mydata=pd.read_stata('grand_crude_oil_converted_19_09_2023.dta')    
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
#Black-Scholes implied volatility functions
def newton_vol_call(S, K, T, C, r, sigma):
    
    #S: spot price
    #K: strike price
    #T: time to maturity  (as per the VIX white paper guidelines)
    #C: Call value
    #r: interest rate
    #sigma: volatility of underlying asset
    
    #The Newton-Raphson method.
    
    d1 = (np.log(S/K) + (0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = (np.log(S/K) - (0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    
    fx = np.exp(-r*T)*(S*si.norm.cdf(d1, 0.0, 1.0) - K*si.norm.cdf(d2, 0.0, 1.0))-C
    
    vega = np.exp(-r*T)*(S*(1/np.sqrt(2*math.pi))*np.exp(-0.5*(d1**2))*(0.5*np.sqrt(T)-(np.log(S/K)/(sigma**2)*np.sqrt(T)))
                         +K*(1/np.sqrt(2*math.pi))*np.exp(-0.5*(d2**2))*(0.5*np.sqrt(T)+(np.log(S/K)/(sigma**2)*np.sqrt(T)))) 
    
    tolerance = 0.000001
    x0 = sigma
    xnew  = x0
    xold = x0 - 1
        
    while abs(xnew - xold) > tolerance:
    
        xold = xnew
        xnew = xnew - (fx/vega)
        
        return abs(xnew)

def newton_vol_put(S, K, T, P, r, sigma):
    
    d1 = (np.log(S/K) + (0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = (np.log(S/K) - (0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    
    fx = np.exp(-r*T)*(K*si.norm.cdf(d2, 0.0, 1.0)-S*si.norm.cdf(d1, 0.0, 1.0))-P
    
    vega = np.exp(-r*T)*(S*(1/np.sqrt(2*math.pi))*np.exp(-0.5*(d1**2))*(0.5*np.sqrt(T)-(np.log(S/K)/(sigma**2)*np.sqrt(T)))
                         +K*(1/np.sqrt(2*math.pi))*np.exp(-0.5*(d2**2))*(0.5*np.sqrt(T)+(np.log(S/K)/(sigma**2)*np.sqrt(T)))) 
    
    tolerance = 0.000001
    x0 = sigma
    xnew  = x0
    xold = x0 - 1
        
    while abs(xnew - xold) > tolerance:
    
        xold = xnew
        xnew = xnew - (fx/vega)
        
        return abs(xnew)
    
#%% 
#loop to facilitate the datewise computation and store the same. 
final_data=pd.DataFrame(columns = ["date", "vix"])
range_data=pd.DataFrame(columns = ["date", "range"])

for x in range(len(list_date)):
    
    #Create the time window.
    #Do not forget to change all the three variables defined just below. 
    exact_days=90                          
    t1=list_date[x]+np.timedelta64(83,'D') 
    t2=list_date[x]+np.timedelta64(97,'D')   
    
    #Select the options expiring in that time window. 
    #Also, make sure that the gap between the underlyig futures expiration date
    #and option expiration date is less than 30 days.
    #Maximum gap allowance depends on the contract specifications. 
    #This condition makes sure that we are using options defined on front-month futures only to compute VIX/SVIX/BSIV etc.   
    filtered_df=mydata.loc[(mydata["option_exp"]>=t1) & (mydata["option_exp"]<=t2) \
                               & (mydata["date1"]==list_date[x]) \
                                   & ((mydata["fut_exp_date"]-mydata["option_exp"])<=np.timedelta64(7,'D'))]  
    
    #Drop observations with zero strikes. They are simply clerical errors and create issues with BSIV estimation. 
    filtered_df=filtered_df.loc[(filtered_df["strike"]>0)]    
        
    #Skip to the next date if no observation found for the selected time window. 
    if len(filtered_df)==0:
        #Drop any VIX NaN value assigned for this date. 
        final_data=final_data.loc[(final_data['date']!=list_date[x]) & (final_data['vix']!=np.nan)]
        df1=pd.DataFrame({'date':list_date[x], 'vix':np.nan}, index=[0])
        final_data=pd.concat([final_data, df1], ignore_index=True)
        continue
    #Proceed with main bloack computation. 
    if len(filtered_df)!=0:  
        #calculate the time to expiration
        filtered_df["time"]=(filtered_df["option_exp"]-list_date[x])
        filtered_df["days"]= filtered_df["time"].dt.days
        #calculate the time to expiration in minutes per year
        filtered_df["minute"]=filtered_df["days"]*(24*60/525600)
        #Drop observations with zero strikes. They are simply clerical errors and create unwarranted issues.  
        filtered_df=filtered_df.loc[(filtered_df["strike"]>0)]
        
        # We don't need to drop any observations based on BSIV in this code. 
        # #Compute BS IV for each option available in the data
        # for index, row in filtered_df.iterrows(): 
        #     S=row['fut_price']
        #     K=row['strike']
        #     T=row['minute']
        #     l=row['price']
        #     r=row['interest_rate']
            
        #     if row['type']=="C":
        #         filtered_df.loc[index,'bsiv']=newton_vol_call(S, K, T, l, r,0.1)
                
        #     if row['type']=="P":
        #         filtered_df.loc[index,'bsiv']=newton_vol_put(S, K, T, l, r,0.1)
        
        # #Drop the options with a very high BS implied volatility. 
        # #Find the lowest "bsiv" in the data and allow for 10% deviation from that.
        # minimum_bsiv=filtered_df["bsiv"].min() 
        # allowed_bsiv=minimum_bsiv+0.10 
        # filtered_df1=filtered_df.loc[(filtered_df["bsiv"]<=allowed_bsiv)]  
        # #This condition needs to be there, because we might have cases with abnormally high BSIV values. 
        
        #We also need to get rid of options with zero or negative prices. 
        #Such observations are simply clerical errors in the data.
        
        filtered_df1=filtered_df
        filtered_df1=filtered_df1.loc[(filtered_df1["price"]>0)] 
        
        if len(filtered_df1)==0:
            #Drop any VIX NaN value assigned for this date. 
            final_data=final_data.loc[(final_data['date']!=list_date[x]) & (final_data['vix']!=np.nan)]
            df1=pd.DataFrame({'date':list_date[x], 'vix':np.nan}, index=[0])
            final_data=pd.concat([final_data, df1], ignore_index=True)
            continue
        
        if len(filtered_df1)!=0:     
            #Find out the number of different expiration dates in the selected sample.
            list_exp_date=np.unique(filtered_df1.option_exp)
            df2=pd.DataFrame({'date':list_date[x], 'range':len(list_exp_date)}, index=[0]) 
            range_data=pd.concat([range_data, df2], ignore_index=True)
            #We will have to deal with multiple sigma cases.
            #The CME group has introduced weekly options even for agricultural commodities. 
            sigma_store=pd.DataFrame(columns = ["time", "sigma_sq"])
            
            for y in range(len(list_exp_date)):
                #Print the current date which is being worked upon. 
                print(list_date[x], list_exp_date[y])
                new_data=filtered_df1.loc[(filtered_df1["option_exp"]==list_exp_date[y])]
                #Extract the underlying futures price. 
                K=new_data.iloc[0,13]
                 
                # #No need to have this condition at all. 
                # #We allow for all the valid observations to be used for VIX/SVIX computation. 
                # #Keep only at-the-money options to compute VIX/SVIX/BSIV etc. 
                # K_max=1.1*K
                # K_min=0.9*K
                # new_data=new_data.loc[(new_data["strike"]>=K_min) & (new_data["strike"]<=K_max)]
                
                #Select out-of-the-money options. 
                new_data1=new_data.loc[(new_data["type"]=="P") & (new_data["strike"]<=K)]
                new_data2=new_data.loc[(new_data["type"]=="C") & (new_data["strike"]>=K)]
                new_data3=new_data1[['strike', 'type','price']]
                new_data4=new_data2[['strike', 'type','price']]
                new_data5=pd.concat([new_data3, new_data4], ignore_index=True)
                new_data6=new_data5.sort_values("strike",ignore_index=True)

                #None or just one out-of-the-money option found. We skip that date and assign NaN to VIX value. 
                if len(new_data6)<=1:
                    #Drop VIX NaN value for this date if previously stored in the final data.
                    final_data=final_data.loc[(final_data['date']!=list_date[x]) & (final_data['vix']!=np.nan)]
                    df1=pd.DataFrame({'date':list_date[x], 'vix':np.nan}, index=[0])
                    final_data=pd.concat([final_data, df1], ignore_index=True)
                    continue
                
                #We have at least two out-of-the-money options to proceed for the main calculation. 
                if len(new_data6)>=2:
                    #Now, we need to compute the contribution made by each strike. 
                    #Step 2 in the CBOE VIX Whitepaper manual. 
                    new_data6['contribution']=0
                    #interest rate 
                    R=new_data.iloc[0,12]
                    #time in minute
                    T=new_data.iloc[0,16]
                    for m in range(len(new_data6)):    
                        #First observation or row. 
                        #Denominator changes for VIX and SVIX. No need to make any other changes. 
                        #Note: K is the underlying futures price. 
                        new_data6.iloc[0,3]=((new_data6.iloc[1,0]-new_data6.iloc[0,0])/(K**2))*math.exp(R*T)*new_data6.iloc[0,2]
                        
                        #Last observation or row. 
                        if m==len(new_data6)-1:
                            new_data6.iloc[m,3]=((new_data6.iloc[m,0]-new_data6.iloc[m-1,0])/(K**2))*math.exp(R*T)*new_data6.iloc[m,2]
                        
                        #All the observations or rows in between. 
                        if ((m>0) and (m<len(new_data6)-1)):
                            new_data6.iloc[m,3]=(((new_data6.iloc[m+1,0]-new_data6.iloc[m-1,0])/2)/(K**2))*math.exp(R*T)*new_data6.iloc[m,2]
                    
                    #We don't need this condition anymore. We had it in the first round of estimation.
                    #Because, we had no information about volume and open interests back then. 
                    #Drop the observation if the contribution is too high!
                    #something is weird with those observations!
                    #new_data6=new_data6.loc[(new_data6["contribution"]<0.01)] 
                    
                    sigma_sq=((2/T)*new_data6.contribution.sum())
                    df3=pd.DataFrame({'time':T, 'sigma_sq':sigma_sq}, index=[0])
                    sigma_store=pd.concat([sigma_store, df3], ignore_index=True)
                    #Sort sigmas based on time. Helps a lot in further computation. 
                    sigma_store=sigma_store.sort_values("time",ignore_index=True)
                    #End of this loop; getting out of this loop and proceed for the VIX computation.
            
            #No need for this condition as the VIX/SVIX computations are in annualized term already.          
            #VIX adjusted for the exact number of days. 
            #sigma_store['vix']=100*np.sqrt(sigma_store['sigma_sq'])*np.sqrt(exact_days/(sigma_store['time']*365))    
            
            sigma_store['vix']=100*np.sqrt(sigma_store['sigma_sq'])
            #Taking mean of all the available vix candidates. 
            vix=sigma_store.vix.mean()            
            #Drop VIX NaN value for this date if previously stored in the final data.
            final_data=final_data.loc[(final_data['date']!=list_date[x]) & (final_data['vix']!=np.nan)]
            df1=pd.DataFrame({'date':list_date[x], 'vix':vix}, index=[0])
            final_data=pd.concat([final_data, df1], ignore_index=True)
                                  
                                    
#done with the first/main loop
#get out of this loop and export the final_data and range_data
                
#%% 
#Change working directory before exporting the data. 
os.getcwd()
os.chdir("M:\\MAHENKS1\\study_stuff\\Commodity_Markets\\PhD_Thesis_Work\\Chapter_1_Volatility\\work_in_progress\\data_analysis\\2023_09_20") 
os.getcwd()  

#export range data
range_data.to_csv("range_data.csv", index = False)

#export final data

#CORN
#file_name="corn_svix_15.xlsx" 
#file_name="corn_svix_30.xlsx" 
#file_name="corn_svix_45.xlsx" 
#file_name="corn_svix_60.xlsx" 
#file_name="corn_svix_90.xlsx" 

#SOYBEAN
#file_name="soybean_svix_15.xlsx" 
#file_name="soybean_svix_30.xlsx" 
#file_name="soybean_svix_45.xlsx" 
#file_name="soybean_svix_60.xlsx" 
#file_name="soybean_svix_90.xlsx" 

#WHEAT 
#file_name="wheat_svix_15.xlsx" 
#file_name="wheat_svix_30.xlsx" 
#file_name="wheat_svix_45.xlsx" 
#file_name="wheat_svix_60.xlsx" 
#file_name="wheat_svix_90.xlsx" 

#Light Sweet Crude Oil 
#file_name="crude_oil_svix_15.xlsx" 
#file_name="crude_oil_svix_30.xlsx" 
#file_name="crude_oil_svix_45.xlsx" 
#file_name="crude_oil_svix_60.xlsx" 
file_name="crude_oil_svix_90.xlsx"

#SOYBEAN OIL
#file_name="soybean_oil_svix_15.xlsx" 
#file_name="soybean_oil_svix_30.xlsx" 
#file_name="soybean_oil_svix_45.xlsx" 
#file_name="soybean_oil_svix_60.xlsx" 
#file_name="soybean_oil_svix_90.xlsx" 

#SOYBEAN MEAL
#file_name="soybean_meal_svix_15.xlsx" 
#file_name="soybean_meal_svix_30.xlsx" 
#file_name="soybean_meal_svix_45.xlsx" 
#file_name="soybean_meal_svix_60.xlsx" 
#file_name="soybean_meal_svix_90.xlsx"

#Henry Hub Natural Gas 
#file_name="natural_gas_svix_15.xlsx" 
#file_name="natural_gas_svix_30.xlsx" 
#file_name="natural_gas_svix_45.xlsx" 
#file_name="natural_gas_svix_60.xlsx" 
#file_name="natural_gas_svix_90.xlsx"

#create the Excel file. 
final_data.to_excel(file_name, index=False)
            
#find out number of nan values stored in the vix column
print(final_data['vix'].isnull().sum())                           
            
#%% The End #%% 




