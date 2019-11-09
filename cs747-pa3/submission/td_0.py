import pdb
import sys
import numpy as np
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
    f=open(data, "r")
    f1=f.readlines()
    f1_exceptlast=f1[:-1]
    f.close()
    total_epochs=1000
    s=0
    r=0
    next_s=0
    next_r=0
    for epoch in range(total_epochs):
        # print("epoch = "+ str(epoch)+ " out of "+str(total_epochs))
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
                if(index==3):
                    s,_,r=line.split()
                    s=int(s)
                    r=float(r)
                else:
                    next_s,_,next_r = line.split()
                    next_s=int(next_s)
                    next_r=float(next_r)
                    V[s]+= alpha*(r+discount_factor*V[next_s]-V[s])
                    s=next_s
                    r=next_r
        
        next_s= f1[-1]
        next_s=int(next_s)
        V[s]+= alpha*(r+discount_factor*V[next_s]-V[s])
    for Vi in V:
        print(Vi)
if __name__ == '__main__':
    main()
