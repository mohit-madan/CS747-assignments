import sys
import numpy as np
import random
import pdb
import signal
#SIGINT handler
def sigint_handler(signal, frame):
    #Do something while breaking
    pdb.set_trace()
    sys.exit(0)

def main():
    # pdb.set_trace()
    instance=sys.argv[2]
    algorithm=sys.argv[4]
    random_seed=int(sys.argv[6])
    epsilon=float(sys.argv[8])
    horizon=int(sys.argv[10])

    np.random.seed(random_seed)
    f=open(instance,"r")
    f1=f.readlines()
    f.close()
    arms=[]
    for i in f1:
        arms.append(float(i.strip()))
    arms_len=len(arms)
    max_reward=max(arms)*horizon
    cum_reward=0
    for time_instance in range(horizon):
        if(np.random.uniform(0,1)<arms[time_instance%arms_len]):
            cum_reward+=1
    regret=round(max_reward-cum_reward,2)
    ret_str=instance+", "+algorithm+", "+ str(random_seed)+", "+str(epsilon)+", "+str(horizon)+", "+str(regret)+"\n"
    save_file=open("outputData.txt","a")
    save_file.write(ret_str)
    save_file.close()

if __name__ == '__main__':
    main()
