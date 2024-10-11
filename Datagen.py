import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
# #Displacement
# X = np.random.rand(10, 4)
# s=X[:, 0]
# v=X[:, 1]
# t=X[:, 2]
# g=X[:, 3]
# x=np.add(s,np.multiply(t,v))
# y=np.negative(np.divide(np.multiply(np.power(t,2),g),2))
# S=np.add(x,y)
# data = np.concatenate((X,S[:, np.newaxis]), axis=1)
# np.savetxt('Data1.txt', data)

# #data gen no dim
# p1=np.divide(S,s)
# p2=np.divide(np.multiply(s,g),np.power(v,2))
# p3=np.divide(np.multiply(v,t),s)
# # p4=-(p2*p3**2)/2 + p3 +1
# data2 = np.concatenate((p2[:, np.newaxis], p3[:, np.newaxis], p1[:, np.newaxis]), axis=1)
# np.savetxt('DataNodim1.txt', data2)

# #Terminal Velocity
# X = np.random.rand(1000, 5)
# m=X[:, 0]
# p=X[:, 1]
# c=X[:, 2]
# a=X[:, 3]
# g=X[:, 4]
# numerator = np.sqrt(np.multiply(2, np.multiply(m, g)))
# denominator = np.sqrt(np.multiply(np.multiply(p, c), a))
# vt = np.divide(numerator,denominator)
# data = np.concatenate((X,vt[:, np.newaxis]), axis=1)
# np.savetxt('Data2.txt', data)

# #data gen no dim
# q1=np.divide(vt,np.multiply(np.power(a,1/4),np.sqrt(g)))
# q2=np.divide(m,np.multiply(p,np.power(a,3/2)))
# q3=c
# q4=np.divide(np.multiply(2,q2),np.power(q1,2))
# data1 = np.concatenate((q1[:, np.newaxis], q2[:, np.newaxis], q3[:, np.newaxis]), axis=1)
# np.savetxt('DataNodim2.txt', data1)

# #Gravitational Potential Enegry
# X = np.random.rand(10, 5)
# m1=X[:, 0]
# m2=X[:, 1]
# r1=X[:, 2]
# r2=X[:, 3]
# g=X[:, 4]

# u=np.subtract(np.divide(np.multiply(g,np.multiply(m2,m1)),r2),np.divide(np.multiply(g,np.multiply(m2,m1)),r1))
# #U=g*m1*m2*(1/r2-1/r1)
# data = np.concatenate((X, u[:, np.newaxis]), axis=1)
# np.savetxt('Data3.txt', data)

# #Data no dim
# p1=np.divide(np.multiply(r2,u),np.multiply(g,np.power(m1,2)))
# p2=np.divide(m2,m1)
# p3=np.divide(r1,r2)
# data1 = np.concatenate((p2[:, np.newaxis], p3[:, np.newaxis], p1[:, np.newaxis]), axis=1)
# np.savetxt('DataNodim3.txt', data1)

# #Darcy_Weisbach Equation for Frictional Pressure Drop
# X = np.random.rand(10, 5)
# f=X[:, 0]
# l=X[:, 1]
# p=X[:, 2]
# v=X[:, 3]
# d=X[:, 4]

# pf=np.divide(np.multiply(np.multiply(f,l),np.multiply(p,np.power(v,2))),np.multiply(2,d))
# Pf=(f*l*p*v*v)/(2*d)
# data = np.concatenate((X,pf[:, np.newaxis]), axis=1)
# np.savetxt('Data4.txt', data)

# #Data no dim
# p1=np.divide(pf,np.multiply(p,np.power(v,2)))
# p2=np.divide(l,d)
# p3=f
# data1 = np.concatenate((p2[:, np.newaxis], p3[:, np.newaxis], p1[:, np.newaxis]), axis=1)
# np.savetxt('DataNodim4.txt', data1)

# #Exponential Growth/Decay
# X = np.random.rand(1000, 6)
# n0=X[:, 0]
# m=X[:, 1]
# g=X[:, 2]
# x=X[:, 3]
# Kb=X[:, 4]
# t=X[:, 5]

# n=np.multiply(n0,np.exp(np.divide(np.multiply(m,np.multiply(g,x)),np.negative(np.multiply(Kb,t)))))
# #N=n0*np.exp((-m*g*x)/(Kb*t))
# data = np.concatenate((X,n[:, np.newaxis]), axis=1)
# np.savetxt('Data5.txt', data)

# #Data no dim
# p1=np.negative(np.divide(np.multiply(Kb,t),np.multiply(m,np.multiply(g,x))))
# p2=n
# p3=n0
# data1 = np.concatenate((p1[:, np.newaxis], p3[:, np.newaxis], p2[:, np.newaxis]), axis=1)
# np.savetxt('DataNodim5.txt', data1)

# #Gravitational Force 
# X = np.random.rand(1000, 4)
# m1=X[:, 0]
# m2=X[:, 1]
# r=X[:, 2]
# g=X[:, 3]

# x=np.multiply(g,np.multiply(m2,m1))
# #y=np.power(r,2)
# f=np.negative(np.divide(x,np.square(r)))
# #f=-(g*m1*m2)/(r**2)
# data = np.concatenate((X, f[:, np.newaxis]), axis=1)
# np.savetxt('Data6.txt', data)

# #Data no dim
# p1=np.divide(np.multiply(np.power(r,2),f),np.multiply(np.power(m1,2),g)) 
# p2=np.divide(m2,m1)
# data1 = np.concatenate((p2[:, np.newaxis], p1[:, np.newaxis]), axis=1)
# np.savetxt('DataNodim6.txt', data1)


with open("UPINN2data.txt", "w") as f:
    # Loop to generate 100 values
    for i in range(11):
        for j in range (3):
        # Calculate x value
            x = i#0 + i *0.5
            
            # Generate a random g value between 0 and 2
            g = j #random.randint(0,20)

            # Calculate ht value based on the given formula
            ht = - np.sin(x)+(g * np.sin(x) * np.cos(x))
            
            # Write the values to the file in the desired format
            f.write(f"{x:.15f} {g:.15f} {ht:.15f}\n")


# data = np.loadtxt('UPINN2Data.txt')

# x0=data[:,0]
# for i in range (3):
#     x1=i
#     truesol=-np.sin(x0)+x1*np.sin(x0)*np.cos(x0)
#     allsol=0.000000000000+(((x1*np.cos(x0))-1)*np.sin(x0))

#     # Create the plot
#     plt.figure(figsize=(10, 6))

#     plt.plot(x0, truesol, label=f'True Hidden Term')
#     plt.plot(x0, allsol, label=f'Equation found by AI-Feynman')
#     #plt.plot(t_out, allsol, label=f'Solution to All Bead')
#     plt.xlabel('Theta')
#     plt.ylabel('Hidden Term')
#     plt.title(f'Rotating Bead Graph_Gamma Value {i}')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(f'UPINN2_AI_feynman_Gamma Value {i}')
#     plt.clf()
