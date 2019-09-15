import sys
import argparse
import pdb
import signal
import numpy as np
from pulp import *

#SIGINT handler
def sigint_handler(signal, frame):
    pdb.set_trace()
    sys.exit(0)

def calc_val(T,R,policy,gamma):
    states,action,_=np.shape(T)
    b=np.array([sum([T[i,policy[i],j]*R[i,policy[i],j] for j in range(states)]) for i in range(states)])
    a=np.eye(states)-np.diag(np.array([gamma*T[i,policy[i],i] for i in range(states)]))-[[gamma*T[i,policy[i],j] if i!=j else 0 for j in range(states)] for i in range(states)]
    V = np.linalg.solve(a,b)
    return V  

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mdp', type=str)
    parser.add_argument('--algorithm', type=str, default='lp')
    args=parser.parse_args()
    f=open(args.mdp,"r")
    f1=f.readlines()
    f.close()

    reward=True
    # storing values in mdp state variables
    for i,v in enumerate(f1):
        if(i==0):
            S=int(v.strip())
        elif(i==1):
            A=int(v.strip())
            R=np.zeros((S,A,S)) # of shape s x a x s_prime
            T=np.zeros((S,A,S)) # of shape s x a x s_prime
            s=0
            a=0
        elif(i<len(f1)-2):
            x=(v.strip()).split('\t')
            for sPrime,val in enumerate(x):
                if(reward):
                    R[s][a][sPrime]=float(val)
                else:
                    T[s][a][sPrime]=float(val)
            a+=1
            if(a==A):
                a=0
                s+=1
            if(s==S):
                s=0
                a=0
                reward=False
        elif(i==len(f1)-2):
            gamma=float(v.strip())
        else:
            mdp_type=v.strip()

    optimal_val=np.zeros(S)
    optimal_action=np.zeros(S, dtype="int")

    if(args.algorithm=='lp'):
        prob= LpProblem("Finding Optimal Policy", LpMinimize)
        V=[]
        for i in range(S):
            x="V_"+str(i)
            x1=LpVariable(x,None,None,LpContinuous)
            V.append(x1)
        prob += lpSum(V) # objective function

        count=1
        for s in range(S):
            for a in range(A):
                prob += V[s] >= lpSum([T[s,a][i]*(R[s,a][i] + gamma*V[i]) for i in range(S)]), "Condition "+str(count)
                count+=1
        # find the optimal solution of the linear inequalities
        prob.solve()
        
        if(LpStatus[prob.status]=="Optimal"):
            for idx,v in enumerate(prob.variables()):
                optimal_val[idx]=v.varValue
                # print(v.name, "=", v.varValue)

        for s in range(S):
            for a in range(A):
                diff = abs(optimal_val[s]-sum([T[s,a,i]*(R[s,a,i]+gamma*optimal_val[i]) for i in range(S)]))
                # print(diff)
                if(diff<1e-6):
                    optimal_action[s]=a
            print(repr(optimal_val[s])+"\t"+str(optimal_action[s]))
    
    # howard's policy iteration
    else:
        init_action=np.random.randint(A,size=S)
        init_val=np.random.rand(S)
        V=calc_val(T,R,init_action,gamma)   
        Q=np.zeros((S,A))
        while(True):
            for s in range(S):
                for a in range(A):
                    Q[s,a]=sum([T[s,a,i]*(R[s,a,i] + gamma*V[i]) for i in range(S)])
            # pdb.set_trace()
            if(np.count_nonzero(init_action-np.argmax(Q,axis=1))==0):
                # print(init_action)
                break
            else:
                init_action=np.argmax(Q,axis=1)
                V=calc_val(T,R,init_action,gamma)
        for s in range(S):
            print(repr(V[s])+"\t"+str(init_action[s]))
        
        pdb.set_trace()

if __name__ == '__main__':
    main()