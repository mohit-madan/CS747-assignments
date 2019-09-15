import pandas as pd
import pdb
import numpy as np
import matplotlib.pyplot as plt
complete_data = pd.read_csv('outputData.txt', delimiter=',', header=None, names=["instance", "algorithm", "seed", "epsilon", "horizon","regret"])
df=pd.DataFrame(complete_data)
file_instance="../instances/i-3.txt"
x=df[df['instance']==file_instance]
y=x[x['algorithm']==" round-robin"]
count=0
regret_mean=np.zeros(7)
for horizon in 50, 200,800, 3200, 12800, 51200, 204800:
	z=y[y['horizon']==horizon]
	regret_mean[count]=np.mean(z['regret'].to_numpy())
	count+=1
plt.xscale("log")
plt.yscale("log")
plt.ylabel("Regret")
plt.xlabel("Horizon")
xs=np.array([50, 200,800, 3200, 12800, 51200, 204800])	

plt.title("i-3: regret vs horizon")
plt.plot(xs,regret_mean, marker='o',label="round-robin")
# plt.show()
# pdb.set_trace()

x=df[df['instance']==file_instance]
x=x[x['algorithm']==" epsilon-greedy"]

regret_mean=np.zeros(7)
for epsilon in 0.002, 0.02, 0.2:
# for epsilon in 0.2:
	count=0
	z=x[x['epsilon']==epsilon]
	for horizon in 50, 200,800, 3200, 12800, 51200, 204800:
		y=z[z['horizon']==horizon]
		regret_mean[count]=np.mean(y['regret'].to_numpy())
		count+=1
	# plt.xscale("log")
	# xs=np.array([50, 200,800, 3200, 12800, 51200, 204800])
	label_str="epsilon-greedy: "+str(epsilon)
	# print(label_str)	
	plt.plot(xs,regret_mean, marker='D',label=label_str)




x=df[df['instance']==file_instance]
x=x[x['algorithm']==" ucb"]

count=0
regret_mean=np.zeros(7)
for horizon in 50, 200,800, 3200, 12800, 51200, 204800:
	y=x[x['horizon']==horizon]
	regret_mean[count]=np.mean(y['regret'].to_numpy())
	count+=1
plt.plot(xs,regret_mean, marker='x',label="ucb")	


x=df[df['instance']==file_instance]
x=x[x['algorithm']==" kl-ucb"]

count=0
regret_mean=np.zeros(7)
for horizon in 50, 200,800, 3200, 12800, 51200, 204800:
	y=x[x['horizon']==horizon]
	regret_mean[count]=np.mean(y['regret'].to_numpy())
	count+=1
plt.plot(xs,regret_mean, marker='*',label="kl-ucb")	


x=df[df['instance']==file_instance]
x=x[x['algorithm']==" thompson-sampling"]

count=0
regret_mean=np.zeros(7)
for horizon in 50, 200,800, 3200, 12800, 51200, 204800:
	y=x[x['horizon']==horizon]
	regret_mean[count]=np.mean(y['regret'].to_numpy())
	count+=1
plt.plot(xs,regret_mean, marker='<',label="thompson-sampling")	
plt.legend()
plt.show()
pdb.set_trace()