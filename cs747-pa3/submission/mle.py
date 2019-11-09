import pdb
import sys
import numpy as np
from collections import deque
#SIGINT handler
def sigint_handler(signal, frame):
    pdb.set_trace()
    sys.exit(0)

def main():
    data=sys.argv[1]
    alpha_init=0.05 #learning rate
    num_states=0
    num_actions=0
    discount_factor=0
    n=1
    f=open(data, "r")
    f1=f.readlines()
    f1_exceptlast=f1[:-1]
    f.close()
    total_epochs=1000
    s=0
    r=0
    next_s=0
    next_r=0
    gamma=0.1
    state=[]
    reward=[]
    for epoch in range(total_epochs):
        print("epoch = "+ str(epoch)+ " out of "+str(total_epochs))
        alpha=alpha_init/(1+epoch)  #decaying learning rate
        for index,line in enumerate(f1_exceptlast):
            if(epoch==0):
                if(index==0):
                    num_states=int(line)
                    V=np.zeros(num_states)

                # elif(index==1):
                #     num_actions=int(line)

                elif(index==2):
                    discount_factor=float(line)

            
            if(index>=3):
                if(index<3+n):
                    s,_,r=line.split()
                    state.append(int(s))
                    reward.append(float(r))
                else:
                    next_s,_,next_r = line.split()
                    state.append(int(next_s))
                    reward.append(float(next_r))
                    g=0
                    for x in range(n):
                        g+=reward[x]*(discount_factor**(x))
                    g+=V[state[n]]*discount_factor**n    
                    V[state[0]]+= alpha*(g-V[state[0]])
                    # pdb.set_trace()
                    state.pop(0)
                    reward.pop(0)
        
        next_s= f1[-1]
        state.append(int(next_s))
        g=0
        for x in range(n):
            g+=reward[x]*discount_factor**(x)
        g+=V[state[n]]*discount_factor**n    
        V[state[0]]+= alpha*(g-V[state[0]])

    for Vi in V:
        print(Vi)
if __name__ == '__main__':
    main()
