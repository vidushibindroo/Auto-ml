#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from math import e,sqrt,sin,cos


# In[2]:


# Identity function: 
def identity(x):
    return x
# Identity function derivative:
def identity_deriv(x):
    return 1


# In[3]:


# Binary step function:
def step(x):
    return 0 if x < 0 else 1


# In[6]:


# Sigmoid/logistic functions with Numpy:
def logistic(x):
    return 1/(1 + np.exp(-x))

# Sigmoid/logistic function derivative:
def logistic_deriv(x):
    return logistic(x)*(1-logistic(x))


# In[7]:


# Tanh/Hyperbolic Tangent Activation Function:
def tanh(x):
    return np.tanh(x)

# Tahn function derivative:
def tanh_deriv(x):
    return 1.0 - np.tanh(x)**2
  
# ArcTan function:
def arctan(x):
    return np.arctan(x)

# ArcTan function derivative:
def arctan_deriv(x):
    return 1/((x**2)+1)


# In[8]:


# Softsign function:
def softsign(x):
    x = abs(x)
    return x/(1+x)

# Softsign function derivative:
def softsign_deriv(x):
    x = abs(x)
    return 1/((1+x))**2


# In[9]:


# Inverse square root unit (ISRU):
def isru(x, a=0.01):
    return x/sqrt(x+(a*(x**2)))

# ISRU derivative:
def isru_deriv(x, a=0.01):
    return (x/sqrt(1+(a*(x**2))))**3


# In[10]:


# Inverse square root LINEAR unit (ISRLU):
def isrlu(x,a=0.01):
    return x/sqrt(1+(a*(x**2))) if x < 0 else x

# ISRLU derivative:
def isrlu_deriv(x,a=0.01):
    return (1/sqrt(1+(a*(x**2))))**3 if x < 0 else 1


# In[11]:


# Square Nonlinearity (SQNL):
def sqnl(x):
    if x < (-2.0): 
        return -1
    elif x < 0.0:
        return x+((x**2.0)/4.0)
    elif x <= 2.0:
        return x-((x**2)/4.0)
    elif x > 2.0:
        return 1

# SQNL derivative:
def sqnl_deriv(x):
  # Note: This function returns two values.
  return (1-(x/2), 1+(x/2))


# In[12]:


# ReLu (Rectified Linear Unit) function:
def relu(x):
    return 0 if x < 0.0 else x
  
# ReLU derivative:
def relu_deriv(x):
    return 0 if x < 0.0 else 1


# In[13]:


# Leaky ReLU:
def leaky(x):
    return x*0.01 if x < 0 else x
      
# Leaky ReLU derivative:
def leaky_deriv(x):
    return 0.01 if x < 0 else 1


# In[14]:


# Parametric rectified linear unit (PReLU):
def prelu(x,a=0.01):
    return x*a if x < 0 else x  
  
# PReLU derivative:
def prelu_deriv(x,a=0.01): 
    return a if x < 0 else 1


# In[15]:


# Exponential linear unit (ELU):
def elu(x,a=0.01):
    return a*(((e)**2)-1) if x <= 0 else x
  
# ELU derivative:
def elu_deriv(x,a=0.01):
    return elu(x,a)+a if x <= 0 else 1


# In[16]:


# SoftPlus
def softplus(x):
    return np.log(1+((e)**x))

# SoftPlus derivative: 
def softplus_deriv(x):
    return 1/(1+((e)**-x))


# In[17]:


# SoftExponential:
def softex(x,a=0.01):
    if a < 0:
        return -((np.log(1-a*(x+a)))/a)
    elif a == 0:
        return x
    elif a > 0:
        return (((e)**(a*x))/a)+a
# SoftExponential derivative:
def softex_deriv(x,a=0.01):
    return 1/(1-a*(a+x)) if a < 0 else (e)**(a*x)


# In[18]:


# Sinusoid:
def sinusoid(x):
    return sin(x)
  
# Sinusoid derivative:
def sinusoid_deriv(x):
    return cos(x) 


# In[19]:


# Gaussian: 
def gaussian(x):
    return (e)**((-x)**2)
   
# Gaussian derivative:
def gaussian_deriv(x):
    return -2*x*(e)**((-x)**2) 


# In[ ]:




