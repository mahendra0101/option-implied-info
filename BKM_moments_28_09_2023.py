#%% 
"""
@author: Mahendra Kumar Singh (mahenks1@iastate.edu) 
@Date: 28 SEP 2023 
"""                
#%% 
#Clear all the variables from the workspace. 
from IPython import get_ipython
get_ipython().magic('reset -sf') 
import numpy as np
import pandas as pd
import math
import os                    
#%%
#Note:- This Python code computes the risk-neutral moments proposed by Bakshi et al. (2003) for a given trading date and allowed time-horizon.   
#%%
#Basic data formatting
print(os.getcwd())
#Locate the directory where all the grand datasets are located. 
os.chdir("M:\\MAHENKS1\\study_stuff\\Commodity_Markets\\PhD_Thesis_Work\\Chapter_1_Volatility\\work_in_progress\\data_analysis\\2023_09_19\\vix_svix_suitable")     
#Change the input data file name over here. 
#mydata=pd.read_stata('grand_corn_converted_19_09_2023.dta')  
#mydata=pd.read_stata('grand_soybean_converted_19_09_2023.dta')  
mydata=pd.read_stata('grand_wheat_converted_19_09_2023.dta')  
#mydata=pd.read_stata('grand_crude_oil_converted_19_09_2023.dta') 
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
final_data=pd.DataFrame(columns = ["date", "var_bkm", "skew_bkm", "kurt_bkm"]) 
range_data=pd.DataFrame(columns = ["date", "range"])

for x in range(len(list_date)):
    
    #Create the time window.
    #Do not forget to change all the three variables defined just below. 
    exact_days=90   
    t1=list_date[x]+np.timedelta64(83,'D') 
    t2=list_date[x]+np.timedelta64(97,'D')        
    
    #Select the options expiring in that time window. 
    #Also, make sure that the gap between the underlyig futures expiration date
    #and option expiration date is less than a fixed time gap (can be obtained from the Barchart futures/options expiration calendar.)
    #This condition makes sure that we are using options defined on front-month futures only to compute VIX/SVIX/BSIV/BKM moments etc.   
    filtered_df=mydata.loc[(mydata["option_exp"]>=t1) & (mydata["option_exp"]<=t2) \
                               & (mydata["date1"]==list_date[x]) \
                                   & ((mydata["fut_exp_date"]-mydata["option_exp"])<=np.timedelta64(24,'D'))]     
        
    #Drop observations with zero strikes. They are simply clerical errors and create multiple issues.  
    filtered_df=filtered_df.loc[(filtered_df["strike"]>0)]       
        
    #Skip to the next date if no observation found for the selected time window. 
    if len(filtered_df)==0:
        #Drop any previous NaN values assigned for this date. 
        final_data=final_data.loc[(final_data['date']!=list_date[x]) & (final_data['var_bkm']!=np.nan) \
                                   & (final_data['skew_bkm']!=np.nan) & (final_data['kurt_bkm']!=np.nan)]
        df1=pd.DataFrame({'date':list_date[x], 'var_bkm':np.nan, 'skew_bkm':np.nan, 'kurt_bkm':np.nan}, index=[0])
        final_data=pd.concat([final_data, df1], ignore_index=True)
        continue
    #Proceed with the main block computation. 
    if len(filtered_df)!=0:  
        #Calculate the time to expiration.
        filtered_df["time"]=(filtered_df["option_exp"]-list_date[x])
        filtered_df["days"]= filtered_df["time"].dt.days
        #Calculate the time to expiration in minutes per year. 
        filtered_df["minute"]=filtered_df["days"]*(24*60/525600)
        #Drop observations with zero strikes. They are simply clerical errors and create multiple issues. 
        filtered_df=filtered_df.loc[(filtered_df["strike"]>0)]

        #We also need to get rid of options with zero or negative prices. 
        #Such observations are simply clerical errors in the data. 
        filtered_df1=filtered_df
        filtered_df1=filtered_df1.loc[(filtered_df1["price"]>0)] 
        
        if len(filtered_df1)==0: 
            #Drop any previous NaN values assigned for this date. 
            final_data=final_data.loc[(final_data['date']!=list_date[x]) & (final_data['var_bkm']!=np.nan) \
                                       & (final_data['skew_bkm']!=np.nan) & (final_data['kurt_bkm']!=np.nan)]
            df1=pd.DataFrame({'date':list_date[x], 'var_bkm':np.nan, 'skew_bkm':np.nan, 'kurt_bkm':np.nan}, index=[0])
            final_data=pd.concat([final_data, df1], ignore_index=True)
            continue
        
        if len(filtered_df1)!=0:     
            #find out the number of different options expiration dates in the selected sample. 
            list_exp_date=np.unique(filtered_df1.option_exp)
            df2=pd.DataFrame({'date':list_date[x], 'range':len(list_exp_date)}, index=[0]) 
            range_data=pd.concat([range_data, df2], ignore_index=True)
            #We will have to deal with multiple sigma cases or multiple options expiration dates. 
            #The CME group has introduced weekly options even for agricultural commodities. 
            #Weekly and are more fancier options are very recent thigs. 
            #These relatively newer options constitute about 10% of the sample in recent years. 
            #We simply can't ignore those from the consideration. 
            
            #A warehouse to store all the BKM moments (var/skew/kurt) for a given trading date and options expiration date.
            #Eventually, we will average out those candidate BKM moments for a given trading date. 
            sigma_store=pd.DataFrame(columns = ["time", "var_bkm", "skew_bkm", "kurt_bkm"])
            
            for y in range(len(list_exp_date)):
                #Print the current date which is being worked upon. 
                print(list_date[x], list_exp_date[y])
                new_data=filtered_df1.loc[(filtered_df1["option_exp"]==list_exp_date[y])]
                
                #Extract the underlying futures price. 
                K=new_data.iloc[0,13]
                                
                #Select out-of-the-money options. 
                new_data1=new_data.loc[(new_data["type"]=="P") & (new_data["strike"]<=K)]
                new_data2=new_data.loc[(new_data["type"]=="C") & (new_data["strike"]>=K)]
                ###############################################
                #Out-of-the-money put options sorted on strikes. 
                new_data3=new_data1[['strike', 'type','price']]
                new_data3=new_data3.sort_values("strike",ignore_index=True)
                ###############################################
                #Out-of-the-money call options sorted on strikes. 
                new_data4=new_data2[['strike', 'type','price']]
                new_data4=new_data4.sort_values("strike",ignore_index=True)
                
                #Note, here comes the major change in the strategy. 
                #VIX and SVIX formulae have the same weight for both the call and put option premiums. 
                #However, BKM moments apply different weights to call and put option premiums. 
                #Check the Bakshi et al. (2003) very carefully for more explanation.                 
                
                #Therefore, we need at-least two out-of-the-money call and out-of-the-money put options to proceed with the BKM estimations. 
                #Otherwise, we skip that date and assign NaN to all the risk-neutral BKM estimates.                    
                if ((len(new_data3)<2) or (len(new_data4)<2)): 
                    #Drop any previous NaN values assigned for this date. 
                    final_data=final_data.loc[(final_data['date']!=list_date[x]) & (final_data['var_bkm']!=np.nan) \
                                               & (final_data['skew_bkm']!=np.nan) & (final_data['kurt_bkm']!=np.nan)]
                    df1=pd.DataFrame({'date':list_date[x], 'var_bkm':np.nan, 'skew_bkm':np.nan, 'kurt_bkm':np.nan}, index=[0])
                    final_data=pd.concat([final_data, df1], ignore_index=True)
                    continue
                
                #We have at least two out-of-the-money call and two out-of-the-money put options to proceed for the main calculation. 
                if ((len(new_data3)>=2) and (len(new_data4)>=2)): 
                    #Now, we need to compute the contribution made by each strike. 
                    #Step 2 in the CBOE VIX Whitepaper manual. 
                    
                    #We need to run the numerical integration scheme twice. 
                    #First, on the out-of-the-money put options. 
                    #Second, on out-of-the-money call options.  
                    #Subsequently, we can compute the V, W, X, and \mu terms. 
                    
                    #interest rate 
                    R=new_data.iloc[0,12]
                    #time in minute
                    T=new_data.iloc[0,16]
                      
                    ##############################################
                    #Computation for Out-of-the-money Put Options.
                    
                    new_data3['v1']=0
                    new_data3['w1']=0
                    new_data3['x1']=0
                    
                    for m in range(len(new_data3)):   
                        
                        ##########################
                        #First observation or row. 
                        #Compute v1 contribution for this strike.
                        new_data3.iloc[0,3]=((new_data3.iloc[1,0]-new_data3.iloc[0,0])/(new_data3.iloc[0,0]**2))*(new_data3.iloc[0,2])*(2*(1+math.log(K/new_data3.iloc[0,0])))
                        
                        #Compute w1 contribution for this strike.
                        new_data3.iloc[0,4]=((new_data3.iloc[1,0]-new_data3.iloc[0,0])/(new_data3.iloc[0,0]**2))*(new_data3.iloc[0,2])*(6*math.log(K/new_data3.iloc[0,0])+3*((math.log(K/new_data3.iloc[0,0]))**2))
                        
                        #Compute x1 contribution for this strike
                        new_data3.iloc[0,5]=((new_data3.iloc[1,0]-new_data3.iloc[0,0])/(new_data3.iloc[0,0]**2))*(new_data3.iloc[0,2])*(12*((math.log(K/new_data3.iloc[0,0]))**2)+4*((math.log(K/new_data3.iloc[0,0]))**3)) 
                                                                                                                                                                                                                                                                                
                        #########################
                        #Last observation or row. 
                        if m==len(new_data3)-1:   
                            #Compute v1 contribution for this strike.
                            new_data3.iloc[m,3]=((new_data3.iloc[m,0]-new_data3.iloc[m-1,0])/(new_data3.iloc[m,0]**2))*(new_data3.iloc[m,2])*(2*(1+math.log(K/new_data3.iloc[m,0])))
                            
                            #Compute w1 contribution for this strike.
                            new_data3.iloc[m,4]=((new_data3.iloc[m,0]-new_data3.iloc[m-1,0])/(new_data3.iloc[m,0]**2))*(new_data3.iloc[m,2])*(6*math.log(K/new_data3.iloc[m,0])+3*((math.log(K/new_data3.iloc[m,0]))**2))
                                                    
                            #Compute x1 contribution for this strike
                            new_data3.iloc[m,5]=((new_data3.iloc[m,0]-new_data3.iloc[m-1,0])/(new_data3.iloc[m,0]**2))*(new_data3.iloc[m,2])*(12*((math.log(K/new_data3.iloc[m,0]))**2)+4*((math.log(K/new_data3.iloc[m,0]))**3))
                            
                        #########################    
                        #All the observations or rows in between. 
                        if ((m>0) and (m<len(new_data3)-1)):
                            
                            #Compute v1 contribution for this strike.
                            new_data3.iloc[m,3]=(((new_data3.iloc[m+1,0]-new_data3.iloc[m-1,0])/2)/(new_data3.iloc[m,0]**2))*(new_data3.iloc[m,2])*(2*(1+math.log(K/new_data3.iloc[m,0])))
                            
                            #Compute w1 contribution for this strike.
                            new_data3.iloc[m,4]=(((new_data3.iloc[m+1,0]-new_data3.iloc[m-1,0])/2)/(new_data3.iloc[m,0]**2))*(new_data3.iloc[m,2])*(6*math.log(K/new_data3.iloc[m,0])+3*((math.log(K/new_data3.iloc[m,0]))**2))
                            
                            #Compute x1 contribution for this strike
                            new_data3.iloc[m,5]=(((new_data3.iloc[m+1,0]-new_data3.iloc[m-1,0])/2)/(new_data3.iloc[m,0]**2))*(new_data3.iloc[m,2])*(12*((math.log(K/new_data3.iloc[m,0]))**2)+4*((math.log(K/new_data3.iloc[m,0]))**3))
                    
                    ###############################################
                    #Computation for Out-of-the-money Call Options.
                    new_data4['v2']=0
                    new_data4['w2']=0
                    new_data4['x2']=0
                    
                    for m in range(len(new_data4)):   
                        ##########################
                        #First observation or row. 

                        #Compute v2 contribution for this strike.
                        new_data4.iloc[0,3]=((new_data4.iloc[1,0]-new_data4.iloc[0,0])/(new_data4.iloc[0,0]**2))*(new_data4.iloc[0,2])*(2*(1-math.log(new_data4.iloc[0,0]/K)))
                        
                        #Compute w2 contribution for this strike.
                        new_data4.iloc[0,4]=((new_data4.iloc[1,0]-new_data4.iloc[0,0])/(new_data4.iloc[0,0]**2))*(new_data4.iloc[0,2])*(6*math.log(new_data4.iloc[0,0]/K)-3*((math.log(new_data4.iloc[0,0])/K)**2))
                        
                        #Compute x2 contribution for this strike
                        new_data4.iloc[0,5]=((new_data4.iloc[1,0]-new_data4.iloc[0,0])/(new_data4.iloc[0,0]**2))*(new_data4.iloc[0,2])*(12*((math.log(new_data4.iloc[0,0])/K)**2)-4*((math.log(new_data4.iloc[0,0])/K)**3))
                        
                        #########################
                        #Last obervation or row. 
                        if m==len(new_data4)-1:
                            
                            #Compute v2 contribution for this strike.
                            new_data4.iloc[m,3]=((new_data4.iloc[m,0]-new_data4.iloc[m-1,0])/(new_data4.iloc[m,0]**2))*(new_data4.iloc[m,2])*(2*(1-math.log(new_data4.iloc[m,0]/K)))
                            
                            #Compute w2 contribution for this strike.
                            new_data4.iloc[m,4]=((new_data4.iloc[m,0]-new_data4.iloc[m-1,0])/(new_data4.iloc[m,0]**2))*(new_data4.iloc[m,2])*(6*math.log(new_data4.iloc[m,0]/K)-3*((math.log(new_data4.iloc[m,0])/K)**2))
                            
                            #Compute x2 contribution for this strike
                            new_data4.iloc[m,5]=((new_data4.iloc[m,0]-new_data4.iloc[m-1,0])/(new_data4.iloc[m,0]**2))*(new_data4.iloc[m,2])*(12*((math.log(new_data4.iloc[m,0])/K)**2)-4*((math.log(new_data4.iloc[m,0])/K)**3))
    
                        #########################    
                        #All the observations or rows in between. 
                        if ((m>0) and (m<len(new_data4)-1)):
                                
                            #Compute v2 contribution for this strike.
                            new_data4.iloc[m,3]=(((new_data4.iloc[m+1,0]-new_data4.iloc[m-1,0])/2)/(new_data4.iloc[m,0]**2))*(new_data4.iloc[m,2])*(2*(1-math.log(new_data4.iloc[m,0]/K)))
                            
                            #Compute w2 contribution for this strike.
                            new_data4.iloc[m,4]=(((new_data4.iloc[m+1,0]-new_data4.iloc[m-1,0])/2)/(new_data4.iloc[m,0]**2))*(new_data4.iloc[m,2])*(6*math.log(new_data4.iloc[m,0]/K)-3*((math.log(new_data4.iloc[m,0])/K)**2))
                            
                            #Compute x2 contribution for this strike
                            new_data4.iloc[m,5]=(((new_data4.iloc[m+1,0]-new_data4.iloc[m-1,0])/2)/(new_data4.iloc[m,0]**2))*(new_data4.iloc[m,2])*(12*((math.log(new_data4.iloc[m,0])/K)**2)-4*((math.log(new_data4.iloc[m,0])/K)**3))
                    
                    #################################
                    #Compute all the requisite terms. 
                    #Out-of-the-money put options.
                    v1=new_data3.v1.sum()
                    w1=new_data3.w1.sum()
                    x1=new_data3.x1.sum()
                    #Out-of-the-money call options. 
                    v2=new_data4.v2.sum()
                    w2=new_data4.w2.sum()
                    x2=new_data4.x2.sum()
                    #Compute the main terms. 
                    V=v1+v2
                    W=w2-w1
                    X=x1+x2
                    #Compute the \mu term. 
                    U=math.exp(R*T)-1-(math.exp(R*T)/2)*V-(math.exp(R*T)/6)*W-(math.exp(R*T)/24)*X
                    #Risk-neutral BKM measures. 
                    var_bkm_raw=(math.exp(R*T)*V-(U**2))/T
                    skew_bkm_raw=(math.exp(R*T)*W-3*U*math.exp(R*T)*V+2*(U**3))/(math.exp(R*T)*V-(U**2))**(3/2)
                    kurt_bkm_raw=(math.exp(R*T)*X-4*U*math.exp(R*T)*W+6*math.exp(R*T)*(U**2)*V-3*(U**4))/(math.exp(R*T)*V-(U**2))**2
                    #################################
                    #Save the BKM measures for the current option expiration date in sigma store.                      
                    df3=pd.DataFrame({'time':T, 'var_bkm':var_bkm_raw, 'skew_bkm':skew_bkm_raw, 'kurt_bkm':kurt_bkm_raw}, index=[0])
                    sigma_store=pd.concat([sigma_store, df3], ignore_index=True)
                    #Sort sigmas based on time. Helps a lot in further computation. 
                    sigma_store=sigma_store.sort_values("time",ignore_index=True)
                    #End of this loop; getting out of this loop and proceed for averaging the BKM measures for particular date.
            
            ###################################
            #Taking mean of all the available BKM candidates. 
            var_bkm=sigma_store.var_bkm.mean()
            skew_bkm=sigma_store.skew_bkm.mean()
            kurt_bkm=sigma_store.kurt_bkm.mean()
            ###################################
            #Drop any previous NaN values stored in the final data for this date.
            final_data=final_data.loc[(final_data['date']!=list_date[x]) & (final_data['var_bkm']!=np.nan) \
                                       & (final_data['skew_bkm']!=np.nan) & (final_data['kurt_bkm']!=np.nan)]
            df1=pd.DataFrame({'date':list_date[x], 'var_bkm':var_bkm, 'skew_bkm':skew_bkm, 'kurt_bkm':kurt_bkm}, index=[0])
            final_data=pd.concat([final_data, df1], ignore_index=True)                                 
                                      
#Done with the first/main loop of this computational exercise. 
#Get out of this loop and export the final_data and range_data. 
                
#%% 
#Change working directory before exporting the data. 
os.getcwd()
os.chdir("M:\\MAHENKS1\\study_stuff\\Commodity_Markets\\PhD_Thesis_Work\\Chapter_1_Volatility\\work_in_progress\\data_analysis\\2023_09_28") 
os.getcwd()  

#export range data
range_data.to_csv("range_data.csv", index = False)

#export final data

#CORN
#file_name="corn_bkm_15.xlsx"
#file_name="corn_bkm_30.xlsx"
#file_name="corn_bkm_45.xlsx"
#file_name="corn_bkm_60.xlsx"
#file_name="corn_bkm_90.xlsx"

#SOYBEAN
#file_name="soybean_bkm_15.xlsx"
#file_name="soybean_bkm_30.xlsx"
#file_name="soybean_bkm_45.xlsx"
#file_name="soybean_bkm_60.xlsx"
#file_name="soybean_bkm_90.xlsx"

#WHEAT 
#file_name="wheat_bkm_15.xlsx"
#file_name="wheat_bkm_30.xlsx"
#file_name="wheat_bkm_45.xlsx"
#file_name="wheat_bkm_60.xlsx"
file_name="wheat_bkm_90.xlsx"

#Crude Oil
#file_name="crude_oil_bkm_15.xlsx"
#file_name="crude_oil_bkm_30.xlsx"
#file_name="crude_oil_bkm_45.xlsx"
#file_name="crude_oil_bkm_60.xlsx"
#file_name="crude_oil_bkm_90.xlsx"

#Soybean Oil
#file_name="soybean_oil_bkm_15.xlsx"
#file_name="soybean_oil_bkm_30.xlsx"
#file_name="soybean_oil_bkm_45.xlsx"
#file_name="soybean_oil_bkm_60.xlsx"
#file_name="soybean_oil_bkm_90.xlsx"

#Soybean Meal
#file_name="soybean_meal_bkm_15.xlsx"
#file_name="soybean_meal_bkm_30.xlsx"
#file_name="soybean_meal_bkm_45.xlsx"
#file_name="soybean_meal_bkm_60.xlsx"
#file_name="soybean_meal_bkm_90.xlsx"

#create the Excel file. 
final_data.to_excel(file_name, index=False)
            
#find out number of nan values stored in the vix column
print(final_data['var_bkm'].isnull().sum())                           
            
#%% The End #%% 

