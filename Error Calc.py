import numpy as np

def f1(x0, x1, x2, x3):
    return 0.000000000000+(x0+(((x2*(x1+(x1-(x3*x2)))))/((0+1)+1))) #0.500000000000*(((x1*(((x1*(-x0))+1)+1))+1)+1)

def f2(x0, x1, x2, x3, x4):
     return 1.414213562373*np.sqrt((x4*((x0/x2)/(x3*x1)))) #0.000000000000+((x1+x1)/(x0*x0))

def f4(x0, x1, x2, x3, x4):
     return 0.500000000000*(x3*(x3*(x2*((x0/x4)*x1)))) #0.500000000000*(x1*x0)

def f5(x0, x1, x2, x3, x4, x5):
     return 0.000000000000+(x0/np.exp((x1*(((x2*x3)/x5)/x4)))) #0.000000000000+(x1/np.exp(((2*(0)+1)/x0)))

def f3(x0, x1, x2, x3, x4):
     return  -1.000000000000*(((x0*(x1*x4))/(-x3))+((x0*(x1*x4))/x2)) #0.000000000000+(x0+(x0/(-x1)))

def f6(x0, x1, x2, x3):
     return  -1.000000000000*(x3*(x1*(x0/(x2*x2)))) #0.000000000000+(-x0)


def g1(x0, x1):
    return 0.500000000000*(((x1*(((x1*(-x0))+1)+1))+1)+1)

def g2(x0, x1):
     return 0.000000000000+((x1+x1)/(x0*x0))

def g4(x0, x1):
     return 0.500000000000*(x1*x0)

def g5(x0, x1):
   return 0.000000000000+(x1/np.exp(((2*(0)+1)/x0)))

def g3(x0, x1):
    
     return  0.000000000000+(x0+(x0/(-x1)))

def g6(x0):
     return  0.000000000000+(-x0)



functions = {
    1: f1,
    2: f2,
    3: f3,
    4: f4,
    5: f5,
    6: f6
}

ndfunctions = {
    1: g1,
    2: g2,
    3: g3,
    4: g4,
    5: g5,
    6: g6
}

# Initialize lists to store the mean error of the last column
results = []
ndresults = []

# Loop through each data file
for i in range(1, 7):
    # Load data from the file
    data = np.loadtxt(f'Data{i}.txt')
    nddata=np.loadtxt(f'DataNodim{i}.txt')
    
    # Get the function for the current file
    func = functions.get(i)
    ndfunc= ndfunctions.get(i)
    
    # Compute the function results and calculate the mean absolute error
    computed_values = np.apply_along_axis(lambda row: func(*row[:-1]), 1, data)
    ndcomputed_values = np.apply_along_axis(lambda row: ndfunc(*row[:-1]), 1, nddata)
    last_column = data[:, -1]
    ndlast_column = nddata[:, -1]
    mean_error = np.mean(np.abs(computed_values - last_column))    
    ndmean_error = np.mean(np.abs(ndcomputed_values - ndlast_column))    

   
 # Store the result
    results.append(f"File: Data{i}.txt: {mean_error}\nFile: DataNodim{i}:{ndmean_error}\n")

# Write results to a text file
with open('mean_errors.txt', 'w') as file:
    file.writelines(results)


