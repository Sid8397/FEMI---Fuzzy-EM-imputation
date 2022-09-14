# FEMI---Fuzzy-EM-imputation
Missing data imputation with Fuzzy EM imputation method for Numerical datasets.

#Fuzzy Expectaion maximization imputaion by Siddharth jain
import numpy as np, numpy.random
from scipy.spatial import distance
import pandas as pd
import numpy as np
import os
from skfuzzy import cmeans
import skfuzzy as fuzz
k = 3
p = 2

###Enter the path in 2-slash formate####
####Example= C:\\Users\\vchav\\Downloads\\Data mining docs\\dataset 1\\Incomplete datasets extracted\\Glass\\Glass_AE_1.xlsx
path=input("Enter the path of your excel dataset file with 2-slash formate: \n")
path03=input("Enter the path of the file to be stored: \n")
path04=input("Enter the path of Original complete dataset for NRMS calculation: \n")
dataset =  pd.read_excel(path,header = None)     
Original_dataset = pd.read_excel(path04 ,header=None)    #full original dataset
Initial_dataset=pd.read_excel(path,header = None) #copy or blank dataset for future use

X = dataset.iloc[:,:].values
Y = Original_dataset.iloc[:,:].values

rows = len(dataset)
columns = len(dataset.columns)
complete_rows, incomplete_rows = [], []
for i in range(rows):
    flag = 0
    for j in range(columns):
        if np.isnan(dataset[j][i]):
            flag = 1
            incomplete_rows.append(i)
            break
        if flag == 0:
             complete_rows.append(i)
     
complete = dataset.dropna()
X = complete
incomplete = dataset[dataset.isna().any(axis=1)]

# print(X)


# Print the number of data and dimension 
n = len(X)
d = len(dataset.columns)
# print(n)
# print(d)
addZeros = np.zeros((n, 1))
X = np.append(X, addZeros, axis=1)
# print("The FCM algorithm: \n")
# print("The training data: \n", X)
# print("\nTotal number of data: ",n)
# print("Total number of features: ",d)
# print("Total number of Clusters: ",k)

# Create an empty array of centers
C = np.zeros((k,d+1))
# print(C)
# print(len(C))

# Randomly initialize the weight matrix
weight = np.round(np.random.dirichlet(np.ones(k),size=n),2)      #random weight for complete dataset
# print("\nThe initial weight: \n", np.round(weight,2))

for it in range(50): # Total number of iterations
    
    # Compute centroid
    for j in range(k):
        denoSum = sum(np.power(weight[:,j],2))
        
        sumMM =0
        for i in range(n):
            mm = np.multiply(np.power(weight[i,j],p),X[i,:])
            sumMM +=mm
        cc = sumMM/denoSum
        C[j] = np.reshape(cc,d+1)
 
    #print("\nUpdating the fuzzy pseudo partition")
    for i in range(n):
        denoSumNext = 0
        for j in range(k):
             denoSumNext += np.power(1/distance.euclidean(C[j,0:d], X[i,0:d]),1/(p-1))
        for j in range(k):
            w = np.power((1/distance.euclidean(C[j,0:d], X[i,0:d])),1/(p-1))/denoSumNext
            weight[i,j] = w 
            weight=np.round(weight,2)
            
# print("\nThe final weights of full dataset: \n", np.round(weight,2))
    
    
# for i in range(n):    
#     cNumber = np.where(weight[i] == np.amax(weight[i]))
#     X[i,d] = cNumber[0]
    
    
# print("\nThe data with cluster number: \n", X)

SSE = 0                 #RMSE = root mean square error
for j in range(k):
    for i in range(n):
        SSE += np.power(weight[i,j],p)*distance.euclidean(C[j,0:d], X[i,0:d])
# print("\nSSE: ",np.round(SSE,4))

weight_I = np.round(np.random.dirichlet(np.ones(k),size=len(dataset)-len(X)),2)    #random weight for incomplete dataset
# print("\nThe initial weight: \n", np.round(weight_I,2))
for it in range(50): # Total number of iterations
    incomplete = dataset[dataset.isna().any(axis=1)]
    NC = incomplete.fillna(0).values
    #print(NC)
    C_I = np.zeros((k,d))
    for it in range(50): # Total number of iterations

        # Compute centroid
        for j in range(k):
            denoSum_I = sum(np.power(weight_I[:,j],2))

            sumMM_I =0
            for i in range(d):
                mm_I = np.multiply(np.power(weight_I[i,j],p),NC[i,:])
                sumMM_I +=mm_I
            cc_I = sumMM_I/denoSum_I
          #  print(cc_I)
         #   print(cc_I.shape)
            C_I[j] = np.reshape(cc_I,(1,d))
    #print(C_I)

      #print("\nUpdating the fuzzy pseudo partition")

    for i in range(d):
        denoSumNext = 0
        for j in range(k):
            denoSumNext += np.power(1/distance.euclidean(C_I[j,0:d], NC[i,0:d]),1/(p-1))
        for j in range(k):
            w_I = np.power((1/distance.euclidean(C_I[j,0:d], NC[i,0:d])),1/(p-1))/denoSumNext
            weight_I[i,j] = w_I  
            weight_I=np.round(weight_I,2)
# print("\nThe final weights for incomplete dataset: \n", np.round(weight_I,2))


weight_R = np.round(np.random.dirichlet(np.ones(k),size=len(dataset)),2)     #weight of entire record
# print("\nThe initial weight of entire dataset: \n", np.round(weight_R,2))
#incomplete = dataset[dataset.isna().any(axis=1)]
complete_R = dataset.fillna(0).values  #entire dataset with imputed 0 at the place of missing values
#print(complete_R)
C_R = np.zeros((k,d))
for it in range(50): # Total number of iterations
    for it in range(50):
       # Compute centroid
        for j in range(k):
            denoSum_R = sum(np.power(weight_R[:,j],2))

            sumMM_R =0
            for i in range(d):
                mm_R = np.multiply(np.power(weight_R[i,j],p),complete_R[i,:])
                sumMM_R +=mm_R
            cc_R = sumMM_R/denoSum_R
              #  print(cc_R)
             #   print(cc_R.shape)
            C_R[j] = np.reshape(cc_R,(1,d))
        #print(C_R)

          #print("\nUpdating the fuzzy pseudo partition")

    for i in range(d):
        denoSumNext = 0
        for j in range(k):
            denoSumNext += np.power(1/distance.euclidean(C_R[j,0:d], complete_R[i,0:d]),1/(p-1))
        for j in range(k):
            w_R = np.power((1/distance.euclidean(C_R[j,0:d], complete_R[i,0:d])),1/(p-1))/denoSumNext
            weight_R[i,j] = w_R  
            weight_R=np.round(weight_R,2)
# print("\nThe final weights of entire dataset: \n", np.round(weight_R,2))



# while true:
mu_I=np.zeros((d,k))         # mean vector for incomplete dataset
k=3
denom=[]
for i in range(k):
    denoSum_mu= sum(weight_I[:,i])
    denom+=[denoSum_mu]
# print(denom)
denom_array=np.asarray(denom)
denom_array.shape
#print(weight_I)
#print(NC)
NC_T=NC.transpose()
tmp_mat = np.matmul(NC_T,weight_I)
tmp_mat
mu_I=tmp_mat/denom_array.reshape((1,k))
# print(mu_I)         


mu_F=np.zeros((d,k))          # mean vector for full dataset
k=3
denom_F=[]
for i in range(k):
    denoSum_mu_F= sum(weight[:,i])
    denom_F+=[denoSum_mu_F]
#print(denom_F)
denom_F_array=np.asarray(denom_F)
denom_F_array.shape
#print(weight)
#print(complete)
complete_T=complete.T.values
tmp_mat_X = np.matmul(complete_T,weight)
tmp_mat_X
mu_F=tmp_mat_X/denom_F_array.reshape((1,k))
# print(mu_F)


mu_R=np.zeros((d,k))  #mean vectore for entire dataset , dimention = Attribute* cluster no.
denom_R=[]
for i in range(k):
    denoSum_mu_R= sum(weight_R[:,i])
    denom_R+=[denoSum_mu_R]
#print(denom_R)
denom_R_array=np.asarray(denom_R)
# print(denom_R_array.shape)
complete_R_T=complete_R.T
tmp_mat_R = np.matmul(complete_R_T,weight_R)
tmp_mat_R
mu_R=tmp_mat_R/denom_R_array.reshape((1,k))
# print(mu_R)
# print(mu_R.shape)
# mu_R[2]


a_col_index=incomplete.columns[~incomplete.isnull().any()].tolist() #complete coloumn index
# print(a_col_index)
m_col_index=incomplete.columns[incomplete.isnull().any()].tolist() #incomplete coloumn index
# print(m_col_index)
m_row_index= incomplete.index.tolist()                            #incomplete row's index
# print(m_row_index)       
# print(incomplete)


def sigma(x,y,z):        #defining variance between any 2 attribute, x=cluster number, y and z are attribute = Sigma_xy
    #sum_sigma=0
    #denom_sigma=[]
    denoSum_sigma= np.sum(np.round((weight_R[:,z]),2))
    num_sigma_1=np.multiply(np.round((complete_R[:,y]-mu_R[y,z]),2),np.round((complete_R[:,x]-mu_R[x,z]),2))
    num_sigma_2=np.sum(np.multiply(np.round((weight_R[:,z]),2),num_sigma_1))
    #np.multiply multiplication does not take more than 2 array at a time, so 2 arrays at time max
    Sigma_pq=num_sigma_2/denoSum_sigma
    return Sigma_pq
# print(np.round(sigma(5,6,1),4))
# print(sigma(5,6,1))
# print(np.round(sigma(8,8,1),4))
sigma_aa=np.zeros((k,len(a_col_index),len(a_col_index))) #sigma_aa= covariance between 2 attributes/avilable and missing
for i in range(0,len(a_col_index)):  #for i=8  len=1
    for j in range(0,len(a_col_index)):   #for j=8   len=1
        for s in range(k):

            sigma_aa[s,i,j]=sigma(a_col_index[i],a_col_index[j],s)
# print(sigma_aa)
# print(sigma_aa[1])

sigma_am=np.zeros((k,len(a_col_index),len(m_col_index)))
for i in range(0,len(a_col_index)):  #for i=8  len=1
    for j in range(0,len(m_col_index)):   #for j=0 to 7 ,  len=8
        for s in range(k):

            sigma_am[s,i,j]=sigma(a_col_index[i],m_col_index[j],s)
# print(sigma_am)
# print(sigma_am[1])
# np.round(sigma_aa,4)
# np.round(sigma_am,4)
sigma_mm=np.zeros((k,len(m_col_index),len(m_col_index)))
for i in range(0,len(m_col_index)):        #for i=o to 7  len=8
    for j in range(0,len(m_col_index)):   
        for s in range(k):

            sigma_mm[s,i,j]=sigma(m_col_index[i],m_col_index[j],s)
# print(sigma_mm)
# sigma_mm.shape

sigma_ma=np.zeros((k,len(m_col_index),len(a_col_index)))
for i in range(0,len(m_col_index)):        #for i=o to 7  len=8
    for j in range(0,len(a_col_index)):    #for j=8 ,   len=1
        for s in range(k):

            sigma_ma[s,i,j]=sigma(m_col_index[i],a_col_index[j],s)
# print(sigma_ma)
# sigma_ma.shape

B=np.zeros((k,len(a_col_index),len(m_col_index)))  #Covariance matrix 
for i in range(k):
    B[i]=np.matmul(np.round(sigma_aa[i],4),np.round(sigma_am[i],4))
# print(B)

#mu_mis is basically mu_R for a_col_index and mu_avl is mu_R for m_col_index, simple reshaping will do it.
mu_mis=np.zeros((k,1,len(m_col_index)))  #mu_mis and mu_avl will be same for caluculation across all the records
for i in range (k):
    for j in range (len(m_col_index)):
        mu_mis[i,0,j]=mu_R[m_col_index[j],i] 
# print(mu_mis)
mu_avl=np.zeros((k,1,len(a_col_index)))
for i in range (k):
    for j in range (len(a_col_index)):
        mu_avl[i,0,j]=mu_R[a_col_index[j],i]
# print(mu_avl)
# mu_mis.shape, mu_avl.shape

x_a=np.zeros((k,1,len(a_col_index)))
comp_d= dataset.fillna(0).values
def x_a(x):       #defining x_a for xth number of record
    x_a_1=np.zeros((k,1,len(a_col_index)))
    for i in range (k):
        for j in range(len(a_col_index)):
            x_a_1[i,0,j]=comp_d[x,a_col_index[j]]
    return x_a_1
# x_a(121), x_a(121).shape, x_m

#MAIN FORMULA OF IMPUTATION (that will generate k dataset according to each cluster)
x_m=np.zeros((k,1,len(m_col_index)))
x_m_list=[]   #list of x_m for all records for all clusters 
#first array of this list will give x_m will be of 1st records for all records
for i in range (len(incomplete)):
    x_m=mu_mis+np.matmul((x_a(m_row_index[i])-mu_avl),B)   #main formula for imputaion
    x_m_list+=[x_m]    
    # print("--",x_m)
# print(x_m_list)   
# print(x_m_list[0])
# x_m_list

def m_raw_index_ind(x):    #missing coloum number for xth record
#defines missing point coloum number for all records seperately
    return [i for i, val in enumerate(x.values) if np.isnan(val)]
# print(m_raw_index_ind(incomplete.loc[88]))  
# print(m_raw_index_ind(incomplete.loc[88])[0] )

# print(incomplete, NC)
NC_K=np.zeros((k,len(NC),d))  #k dimentional incomplete dataset with 0 replaced at the place missing values
for m in range(k):
    for i in range (len(NC)):
        for j in range (d):
            NC_K[m,i,j]=NC[i,j]
# print(NC_K, NC_K.shape)

x_m_matrix=np.delete(NC,a_col_index,1)    #matrix of NC with only coloumns which has one or more missing values
#we'll delete avilable col for operational purpose and add it latter on
# print(a_col_index, x_m_matrix , incomplete, x_m_matrix.shape)

x_m_matrix_kD=np.zeros((k,len(NC),len(m_col_index)))   #k dimentional x_m_matrix , just copied same in k dimention
for i in range (k):
    # for j in range (len(NC)):
    #     for m in range (len(m_col_index)):
    x_m_matrix_kD[i,:,:]=x_m_matrix[:,:]
# print(x_m_matrix_kD.shape,x_m_matrix_kD)

#now lets update matrix with out imputed values only at ther place of NaN
for i in range(k):
    
         for j in range(len(NC)):
         
            for t in (m_raw_index_ind(incomplete.loc[incomplete.index[j]])):
                # print(i,j,t)
                # print(x_m_matrix_kD.shape)
                x_m_matrix_kD[i,j,t]=x_m_list[j][i][0,t]    #[row number ] [cluster number] [position of point=[0,coloumn]]
# print(x_m_matrix_kD.shape , x_m_matrix_kD)   

NC_a_kd=np.zeros((k,len(NC),len(a_col_index)))  #printing NC_a for k clusters
for i in range (k):
    NC_a_kd[i,:,:]=NC[:,a_col_index]
# print(NC_a_kd)
D_I_K=np.zeros((k,len(NC),d))  #represents final imputed dataset for all different cluster dimention = k*len*attribute
for i in range(k):
    for j in range (len(a_col_index)):
        D_I_K[i]=np.insert(x_m_matrix_kD[i],a_col_index[j],NC_a_kd[i].reshape((1,len(NC))),axis=1) 
    # print("__",D_I_K)   
D_I_K=np.round(D_I_K,10)    #rounded up upto 10th decimal for terminating criteria Epsalone<10^-10
# print(D_I_K.shape, D_I_K)
# np.round(D_I_K[0,0,1],12)

complete_R_K=np.zeros((k,len(dataset),d))    #copying complete dataset for k dimentions
for i in range (k):
    complete_R_K[i,:,:]=complete_R[:,:]
# print(complete_R_K.shape, complete_R_K)

#now replacing m_row_index record with our imputed D_I_K records
for i in range (k):
    for j in range (len(m_row_index)):
        complete_R_K[i,m_row_index[j],:]=D_I_K[i,j,:]
# print(complete_R_K.shape, complete_R_K)     #intially imputed dataset- 1st imputaion
# print(complete_R_K[0,88,:])
# complete_R_K[0]  
#######        First imputaion ends hear (it took 1 input and gave 3 output     #################



#########  now in next while loop we take 3 input from above and gives 3 output as well    ############
#########  we have to start apply termination criteria from here with while loop  ############
########   output of this loop will be repeated until 2 cosicutive ansers are same    ###########
#########    for that we simply round dataset upto 10digit accuracy for epsalon to be 10^-10 #######



complete_sid=np.zeros((k,len(dataset),d))
prev_complete_R_K=None
count=0
while (not np.array_equal(prev_complete_R_K,complete_R_K)):
    for sj in range(k):        #complete_R=complete_R_K[sj] , and we'll have to define new NC(not complete dataset)
        mu_R=np.zeros((d,k))  #mean vectore for entire dataset , dimention = Attribute* cluster no.
        denom_R=[]
        for i in range(k):
            denoSum_mu_R= sum(weight_R[:,i])
            denom_R+=[denoSum_mu_R]
        #print(denom_R)
        denom_R_array=np.asarray(denom_R)
        # print(denom_R_array.shape)
        complete_R_T=complete_R_K[sj].T                   #defining datast  derived from 1st time
        tmp_mat_R = np.matmul(complete_R_T,weight_R)
        tmp_mat_R
        mu_R=tmp_mat_R/denom_R_array.reshape((1,k))
        # print(mu_R)
        # print(mu_R.shape)
        # mu_R[2]

    #######we use pre defined index only#################
        # a_col_index=incomplete.columns[~incomplete.isnull().any()].tolist() #complete coloumn index
        # # print(a_col_index)
        # m_col_index=incomplete.columns[incomplete.isnull().any()].tolist() #incomplete coloumn index
        # # print(m_col_index)
        # m_row_index= incomplete.index.tolist()                            #incomplete row's index
        # # print(m_row_index)       
        # # print(incomplete)
    #########################################

        def sigma(x,y,z):        #defining variance between any 2 attribute, x=cluster number, y and z are attribute = Sigma_xy
            #sum_sigma=0
            #denom_sigma=[]
            denoSum_sigma= np.sum(np.round((weight_R[:,z]),2))
            num_sigma_1=np.multiply(np.round((complete_R_K[sj][:,y]-mu_R[y,z]),2),np.round((complete_R_K[sj][:,x]-mu_R[x,z]),2))
            num_sigma_2=np.sum(np.multiply(np.round((weight_R[:,z]),2),num_sigma_1))
            #np.multiply multiplication does not take more than 2 array at a time, so 2 arrays at time max
            Sigma_pq=num_sigma_2/denoSum_sigma
            return Sigma_pq
        # print(np.round(sigma(5,6,1),4))
        # print(sigma(5,6,1))
        # print(np.round(sigma(8,8,1),4))
        sigma_aa=np.zeros((k,len(a_col_index),len(a_col_index))) #sigma_aa= covariance between 2 attributes/avilable and missing
        for i in range(0,len(a_col_index)):  #for i=8  len=1
            for j in range(0,len(a_col_index)):   #for j=8   len=1
                for s in range(k):

                    sigma_aa[s,i,j]=sigma(a_col_index[i],a_col_index[j],s)
        # print(sigma_aa)
        # print(sigma_aa[1])

        sigma_am=np.zeros((k,len(a_col_index),len(m_col_index)))
        for i in range(0,len(a_col_index)):  #for i=8  len=1
            for j in range(0,len(m_col_index)):   #for j=0 to 7 ,  len=8
                for s in range(k):

                    sigma_am[s,i,j]=sigma(a_col_index[i],m_col_index[j],s)
        # print(sigma_am)
        # print(sigma_am[1])
        # np.round(sigma_aa,4)
        # np.round(sigma_am,4)
        sigma_mm=np.zeros((k,len(m_col_index),len(m_col_index)))
        for i in range(0,len(m_col_index)):        #for i=o to 7  len=8
            for j in range(0,len(m_col_index)):   
                for s in range(k):

                    sigma_mm[s,i,j]=sigma(m_col_index[i],m_col_index[j],s)
        # print(sigma_mm)
        # sigma_mm.shape

        sigma_ma=np.zeros((k,len(m_col_index),len(a_col_index)))
        for i in range(0,len(m_col_index)):        #for i=o to 7  len=8
            for j in range(0,len(a_col_index)):    #for j=8 ,   len=1
                for s in range(k):

                    sigma_ma[s,i,j]=sigma(m_col_index[i],a_col_index[j],s)
        # print(sigma_ma)
        # sigma_ma.shape

        B=np.zeros((k,len(a_col_index),len(m_col_index)))  #Covariance matrix 
        for i in range(k):
            B[i]=np.matmul(np.round(sigma_aa[i],4),np.round(sigma_am[i],4))
        # print(B)

        #mu_mis is basically mu_R for a_col_index and mu_avl is mu_R for m_col_index, simple reshaping will do it.
        mu_mis=np.zeros((k,1,len(m_col_index)))  #mu_mis and mu_avl will be same for caluculation across all the records
        for i in range (k):
            for j in range (len(m_col_index)):
                mu_mis[i,0,j]=mu_R[m_col_index[j],i] 
        # print(mu_mis)
        mu_avl=np.zeros((k,1,len(a_col_index)))
        for i in range (k):
            for j in range (len(a_col_index)):
                mu_avl[i,0,j]=mu_R[a_col_index[j],i]
        # print(mu_avl)
        # mu_mis.shape, mu_avl.shape

        x_a=np.zeros((k,1,len(a_col_index)))
        comp_d= dataset.fillna(0).values
        def x_a(x):       #defining x_a for xth number of record
            x_a_1=np.zeros((k,1,len(a_col_index)))
            for i in range (k):
                for j in range(len(a_col_index)):
                    x_a_1[i,0,j]=comp_d[x,a_col_index[j]]
            return x_a_1
        # x_a(121), x_a(121).shape, x_m

        #MAIN FORMULA OF IMPUTATION (that will generate k dataset according to each cluster)
        x_m=np.zeros((k,1,len(m_col_index)))
        x_m_list=[]   #list of x_m for all records for all clusters 
        #first array of this list will give x_m will be of 1st records for all records
        for i in range (len(incomplete)):
            x_m=mu_mis+np.matmul((x_a(m_row_index[i])-mu_avl),B)   #main formula for imputaion
            x_m_list+=[x_m]    
            # print("--",x_m)
        # print(x_m_list)   
        # print(x_m_list[0])
        # x_m_list

    #######NO NEED TO DEFINE m_raw_index_ind as we only use previously deruived missing coloum inderx for ind records
    #     def m_raw_index_ind(x):    #missing coloum number for xth record
    #     #defines missing point coloum number for all records seperately
    #         return [i for i, val in enumerate(x.values) if np.isnan(val)]
    #     # print(m_raw_index_ind(incomplete.loc[88]))  
    #     # print(m_raw_index_ind(incomplete.loc[88])[0] )
        #######################################################################################################
    #we'll have to define differnt NC for each dataset, as new imputed dataset won't have 0 at the place of  missing value
        NC_new=np.zeros((len(NC),d))
        for pq in range (len(NC)):
            NC_new[pq,:]=complete_R_K[sj][incomplete.index[pq],:]
        # print(incomplete, NC)
        NC_K=np.zeros((k,len(NC),d))  #k dimentional incomplete dataset with 0 replaced at the place missing values
        for m in range(k):
            for i in range (len(NC)):
                for j in range (d):
                    NC_K[m,i,j]=NC_new[i,j]
        # print(NC_K, NC_K.shape)

        x_m_matrix=np.delete(NC_new,a_col_index,1)    #matrix of NC with only coloumns which has one or more missing values
        #we'll delete avilable col for operational purpose and add it latter on
        # print(a_col_index, x_m_matrix , incomplete, x_m_matrix.shape)

        x_m_matrix_kD=np.zeros((k,len(NC),len(m_col_index)))   #k dimentional x_m_matrix , just copied same in k dimention
        for i in range (k):
            # for j in range (len(NC)):
            #     for m in range (len(m_col_index)):
            x_m_matrix_kD[i,:,:]=x_m_matrix[:,:]
        # print(x_m_matrix_kD.shape,x_m_matrix_kD)

        #now lets update matrix with out imputed values only at ther place of NaN
        for i in range(k):

                 for j in range(len(NC)):

                    for t in (m_raw_index_ind(incomplete.loc[incomplete.index[j]])):
                        x_m_matrix_kD[i,j,t]=x_m_list[j][i][0,t]    #[row number ] [cluster number] [position of point=[0,coloumn]]
        # print(x_m_matrix_kD.shape , x_m_matrix_kD)   

        NC_a_kd=np.zeros((k,len(NC),len(a_col_index)))  #printing NC_a for k clusters
        for i in range (k):
            NC_a_kd[i,:,:]=NC_new[:,a_col_index]
        # print(NC_a_kd)
        D_I_K=np.zeros((k,len(NC),d))  #represents final imputed dataset for all different cluster dimention = k*len*attribute
        for i in range(k):
            for j in range (len(a_col_index)):
                D_I_K[i]=np.insert(x_m_matrix_kD[i],a_col_index[j],NC_a_kd[i].reshape((1,len(NC))),axis=1)
                # print(D_I_K[i])
            # print("__",D_I_K)   
        D_I_K=np.round(D_I_K,10)    #rounded up upto 10th decimal for terminating criteria Epsalone<10^-10
        # print(D_I_K.shape, D_I_K)
        # np.round(D_I_K[0,0,1],12)

        complete_R_K_2=np.zeros((k,len(dataset),d))    #copying complete dataset for k dimentions
        for i in range (k):
            complete_R_K_2[i,:,:]=complete_R_K[sj][:,:]
        # print(complete_R_K.shape, complete_R_K)

        #now replacing m_row_index record with our imputed D_I_K records
        # complete_sid=np.zeros((k,len(dataset),d))
        # complete_R_K_2=np.zeros((k,len(dataset),d))
        for i in range (k):
            for j in range (len(m_row_index)):
                # print("==", D_I_K[i,j,:].shape)
                complete_R_K_2[i,m_row_index[j],:]=D_I_K[i,j,:]
                # print(complete_sid.shape, complete_R_K_2[i,m_row_index[j],:])
            # print("-------",complete_R_K_2[i])
            complete_sid[i]=complete_R_K_2[i] 
            # print(complete_sid[i,m_row_index[j],:])
        # complete_R_K=complete_sid
    # print(complete_sid.shape,complete_sid)
    prev_complete_R_K = complete_R_K
    complete_R_K=complete_sid
    count+=1
    # print(count)
# print(complete_R_K.shape, complete_R_K)
# print(complete_R_K[0,88,:],dataset.loc[88])

complete_IK=np.zeros((k,len(NC),d))
for i in range(k):
    for j in range(len(NC)):
        complete_IK[i,j,:]=complete_R_K[i,m_row_index[j],:]
# print(complete_IK.shape,complete_IK)

weight_IK=np.zeros((len(NC),k))
for i in range(len(NC)):
    weight_IK[i,:]=weight_R[m_row_index[i],:]
# print(weight_IK)

weighted_comp_IK=np.zeros((k,len(NC),d))
for i in range(k):
    for j in range(len(NC)):
        weighted_comp_IK[i,j,:]=complete_IK[i,j,:]*weight_IK[j,i]
# print(weighted_comp_IK.shape,weighted_comp_IK)

weighted_comp_comb=np.zeros((len(NC),d))
for i in range(k):
    weighted_comp_comb+=weighted_comp_IK[i]
    

for i in range(len(NC)):
    dataset.iloc[m_row_index[i]]=dataset.iloc[m_row_index[i]].fillna(pd.Series(weighted_comp_comb[i]))
Imputed_dataset=dataset
# print("Intial dataset: \n", Original_dataset)
# print("Imputed dataset: \n",Imputed_dataset)

# print(Original_dataset.iloc[m_row_index[0]],Imputed_dataset.iloc[m_row_index[0]])


Imputed_dataset.to_excel(path03,header=False, index=False) 

Z = Imputed_dataset.iloc[:,:].values

deno_sum_nrms=0
for i in range(d):
    for j in range((len(dataset))):
        deno_sum_nrms01= np.power(Y[j,i],2)
        # print(deno_sum_nrms)
        deno_sum_nrms+=deno_sum_nrms01
deno_sum_nrms=np.sqrt(deno_sum_nrms)
# print(deno_sum_nrms)


num_sum_nrms=0
for i in range (d):
    for j in range(len(dataset)):
        num_sum_nrms01=np.power((Y[j,i]-Z[j,i]),2)
        num_sum_nrms+=num_sum_nrms01
num_sum_nrms=np.sqrt(num_sum_nrms)
# print(num_sum_nrms)

NRMS=np.round(num_sum_nrms/deno_sum_nrms,6)
# NMRS=np.round(NRMS,4)
# print(NRMS)

# NRMS_list+=[NRMS]

print("NRMS: ",NRMS)
