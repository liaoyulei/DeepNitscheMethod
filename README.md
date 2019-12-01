# Deep Nitsche Method: Deep Ritz Method with Essential Boundary Conditions  

Code require Python >= 3.6 and TensorFlow >= 1.13.  
 
Directory `mixed` contains code for the first two dimensional example.  
Directory `Ldomain` contains code for the second two dimensional example.  
Directory `coef` contains code for the third two dimensional example.  
Directory `Dirichlet` contains code for high dimensional examples.

Each directory contains file `resnet_nitsche.py`, `resnet_ritz.py`, `resnet_ls.py`, corresponding to three deep learning-based methods. The first configuration means certain equation, and the second configuration equals to $\beta$. e.g. Run `python3 resnet_nitsche.py 1 500`.  
