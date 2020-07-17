import numpy as np


#Confusion Matrix
def conf_mat(y_true,y_pred):
    tp,fp,tn,fn=0,0,0,0
    for i in range(len(y_pred)):
        if(y_pred[i]==y_true[i] and y_pred[i]==1):
            tp+=1
        elif(y_pred[i]==y_true[i] and y_pred[i]==0):
            tn+=1
        elif(y_pred[i]!=y_true[i] and y_pred[i]==1):
            fp+=1
        elif(y_pred[i]!=y_true[i] and y_pred[i]==0):
            fn+=1
        
    mat=np.array([[tn, fp], [fn, tp]])
    return mat

#F1 score
def f1(y_true,y_pred):
    mat=conf_mat(y_true,y_pred)
   #Precision = TruePositives / (TruePositives + FalsePositives)
    p=(mat[1][1]/(mat[1][1]+mat[0][1]))
    #Recall = TruePositives / (TruePositives + FalseNegatives)
    r=mat[1][1]/(mat[1][1]+mat[1][0])
    f1_score=(2*p*r)/(p+r)
    return f1_score

#F2 score
def f2(y_true,y_pred):
    mat=conf_mat(y_true,y_pred)
    #Precision = TruePositives / (TruePositives + FalsePositives)
    p=(mat[1][1]/(mat[1][1]+mat[0][1]))
    #Recall = TruePositives / (TruePositives + FalseNegatives)
    r=mat[1][1]/(mat[1][1]+mat[1][0])
    f2_score=(5*p*r)/(4*p+r)
    return f2_score

#F0.5 score
def f_half(y_true,y_pred):
    mat=conf_mat(y_true,y_pred)
    #Precision = TruePositives / (TruePositives + FalsePositives)
    p=(mat[1][1]/(mat[1][1]+mat[0][1]))
    #Recall = TruePositives / (TruePositives + FalseNegatives)
    r=mat[1][1]/(mat[1][1]+mat[1][0])
    #F0.5-Measure = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
    f_half=(1.25*p*r)/(0.25*p+r)
    return f_half
    