#%% 
"""
@author: Mahendra Kumar Singh (mahenks1@iastate.edu) 
@Date: 26 SEP 2023 
"""    
#%% 
#Clear all the variables from the workspace. 
from IPython import get_ipython
get_ipython().magic('reset -sf') 
import numpy as np
import pandas as pd
import os                      
#%%
#Note:- This code computes average BSIV using the moneyness measure. 
#Basically, we select two at-the-money put options (ATMP) and two at-the-money call options (ATMC). 
#Then, we simply take the average of BSIV (implied volatility) of all four ATMP and ATMC options.  
#%%
#Basic data formatting
print(os.getcwd())
#Locate the directory where all the price converted datasets (with BSIV) are located. 
os.chdir("M:\\MAHENKS1\\study_stuff\\Commodity_Markets\\PhD_Thesis_Work\\Chapter_1_Volatility\\work_in_progress\\data_analysis\\2023_09_19")     
#Change the input data file name over here. 
#mydata=pd.read_stata('corn_converted_price.dta')  
#mydata=pd.read_stata('soybean_converted_price.dta')  
#mydata=pd.read_stata('wheat_converted_price.dta')  
mydata=pd.read_stata('crude_oil_converted_price.dta') 
print(os.getcwd())  
print(mydata.head())
print(mydata.tail())
print(type(mydata))  #print the data type imported. 
print(list(mydata))  #print all the variable names in the dataframe
print(type(mydata.date))
list_date=np.unique(mydata.date1)   #list of unique dates in the data
print(list_date)
print(type(list_date))        
#%% 
#loop to facilitate the datewise computation and store the same. 
final_data=pd.DataFrame(columns = ["date", "avg_bsiv"])

for x in range(len(list_date)):
    
    #Create the time window.
    #Do not forget to change all the three variables defined just below. 
    exact_days=90              
    t1=list_date[x]+np.timedelta64(83,'D') 
    t2=list_date[x]+np.timedelta64(97,'D')        
    
    #Print the working date! 
    print(list_date[x])
    
    #Select the options expiring in that time window. 
    #Also, make sure that the gap between the underlyig futures expiration date
    #and option expiration date is less than a fixed number of days.
    #One can use the futures and options expiration calendar to find out the allowed fixed number of days. 
    #This condition makes sure that we are using options defined on front-month futures only to compute VIX/SVIX/BSIV/Smirk etc.
    #Or, any other option-implied moments. 
    filtered_df=mydata.loc[(mydata["option_exp"]>=t1) & (mydata["option_exp"]<=t2) \
                               & (mydata["date1"]==list_date[x]) \
                                   & ((mydata["fut_exp_date"]-mydata["option_exp"])<=np.timedelta64(7,'D'))]     
        
    #Basic data cleaning: non-zero american price, strike, eu_price, bsiv.  
    filtered_df=filtered_df.loc[(filtered_df["price"]>0)]       
    filtered_df=filtered_df.loc[(filtered_df["strike"]>0)] 
    filtered_df=filtered_df.loc[(filtered_df["eu_price"]>0)] 
    filtered_df=filtered_df.loc[(filtered_df["bsiv"]>0)] 
        
    #Skip to the next date if no observation found for the selected time window. 
    if len(filtered_df)==0:
        #Drop any smirk = NaN value (previously) assigned for this date. 
        final_data=final_data.loc[(final_data['date']!=list_date[x]) & (final_data['avg_bsiv']!=np.nan)]
        df1=pd.DataFrame({'date':list_date[x], 'avg_bsiv':np.nan}, index=[0])
        final_data=pd.concat([final_data, df1], ignore_index=True)  
        continue
    
    #Proceed with main block computation. 
    if len(filtered_df)!=0:  
        
        #Generate the moneyness value. 
        filtered_df["money"]=filtered_df["strike"]/filtered_df["fut_price"]        
        
        ###################################
        #Select at-the-money put option. 
        new_data1=filtered_df.loc[(filtered_df["money"]>=0.95) & (filtered_df["money"]<=1.05) & (filtered_df["type"]=="P")]
        
        #Skip to the next date if no observation found for the above condition. 
        if len(new_data1)==0:
            #Drop any avg_bsiv = NaN value (previously) assigned for this date. 
            final_data=final_data.loc[(final_data['date']!=list_date[x]) & (final_data['avg_bsiv']!=np.nan)]
            df1=pd.DataFrame({'date':list_date[x], 'avg_bsiv':np.nan}, index=[0])
            final_data=pd.concat([final_data, df1], ignore_index=True)  
            continue
        
        if len(new_data1)!=0:
            #We need to find out the options with moneyness closest to 1.00.
            #In fact, we are looking for two closest deviation values. 
            new_data1["deviation"]=abs(new_data1["money"]-1.00)  
            #Find the options with the minimum values of the deviation. 
            #Note: We can have more than one options (probably with different maturity dates) with 
            #the minimum deviation value. 
            min_dev_put=new_data1['deviation'].min()
            #Create a temporary dataframe to find the second-lowest deviation value. 
            temp_data_put=new_data1.loc[(new_data1['deviation']!=min_dev_put)]
            #Second-lowest deviation value. 
            min_dev_put_second=temp_data_put['deviation'].min()
            #Average of the Black-Scholes implied volatilities of all the put options with the minimum deviation value. 
            new_data2=new_data1.loc[(new_data1["deviation"]==min_dev_put) | (new_data1["deviation"]==min_dev_put_second)]
            #Final term of the interest. 
            vol_atmp=new_data2["bsiv"].mean()
        
        ###################################
        #Select at-the-money call option.   
        new_data3=filtered_df.loc[(filtered_df["money"]>=0.95) & (filtered_df["money"]<=1.05) & (filtered_df["type"]=="C")]
        
        #Skip to the next date if no observation found for the above condition. 
        if len(new_data3)==0:
            #Drop any avg_bsiv = NaN value (previously) assigned for this date. 
            final_data=final_data.loc[(final_data['date']!=list_date[x]) & (final_data['avg_bsiv']!=np.nan)]
            df1=pd.DataFrame({'date':list_date[x], 'avg_bsiv':np.nan}, index=[0])
            final_data=pd.concat([final_data, df1], ignore_index=True)  
            continue
        
        if len(new_data3)!=0:
            #We need to find out the options with moneyness closest to 1.00. 
            new_data3["deviation"]=abs(new_data3["money"]-1.00)  
            #Find the options with the minimum values of the deviation. 
            #Note: We can have more than one options (probably with different maturity dates) with 
            #the minimum deviation value. 
            min_dev_call=new_data3['deviation'].min()
            #Created a temporary dataframe to find the second-lowest deviation value. 
            temp_data_call=new_data3.loc[(new_data3['deviation']!=min_dev_call)]
            #Second-lowest deviation value. 
            min_dev_call_second=temp_data_call['deviation'].min()
            #Average of the Black-Scholes implied volatilities of all the call options with the two-lowest deviation values. 
            new_data4=new_data3.loc[(new_data3["deviation"]==min_dev_call) | (new_data3["deviation"]==min_dev_call_second)]
            #Final term of the interest. 
            vol_atmc=new_data4["bsiv"].mean()  
        
        ###################################
        #Calculate the Smirk value and store it. 
        avg_bsiv_value=((vol_atmp+vol_atmc)*100)/2 
        #Drop any previously stored NaN value. 
        final_data=final_data.loc[(final_data['date']!=list_date[x]) & (final_data['avg_bsiv']!=np.nan)]
        df1=pd.DataFrame({'date':list_date[x], 'avg_bsiv':avg_bsiv_value}, index=[0])
        final_data=pd.concat([final_data, df1], ignore_index=True)  
                                  
                                      
#done with the first/main loop
#get out of this loop and export the final_data. 
                
#%% 
#Change working directory before exporting the data. 
os.getcwd()
os.chdir("M:\\MAHENKS1\\study_stuff\\Commodity_Markets\\PhD_Thesis_Work\\Chapter_1_Volatility\\work_in_progress\\data_analysis\\2023_09_26") 
os.getcwd()  

#export final data

#CORN
#file_name="corn_bsiv_15.xlsx"
#file_name="corn_bsiv_30.xlsx"
#file_name="corn_bsiv_45.xlsx"
#file_name="corn_bsiv_60.xlsx"
#file_name="corn_bsiv_90.xlsx"

#SOYBEAN
#file_name="soybean_bsiv_15.xlsx"
#file_name="soybean_bsiv_30.xlsx"
#file_name="soybean_bsiv_45.xlsx"
#file_name="soybean_bsiv_60.xlsx"
#file_name="soybean_bsiv_90.xlsx"

#WHEAT 
#file_name="wheat_bsiv_15.xlsx"
#file_name="wheat_bsiv_30.xlsx"
#file_name="wheat_bsiv_45.xlsx"
#file_name="wheat_bsiv_60.xlsx"
#file_name="wheat_bsiv_90.xlsx"

#Crude Oil
#file_name="crude_oil_bsiv_15.xlsx"
#file_name="crude_oil_bsiv_30.xlsx"
#file_name="crude_oil_bsiv_45.xlsx"
#file_name="crude_oil_bsiv_60.xlsx"
file_name="crude_oil_bsiv_90.xlsx"

#create the Excel file. 
final_data.to_excel(file_name, index=False)
            
#find out number of nan values stored in the vix column
print(final_data['avg_bsiv'].isnull().sum())                           

            
#%% The End #%% 



