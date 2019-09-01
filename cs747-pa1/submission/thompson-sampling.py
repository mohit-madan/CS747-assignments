import sys
import numpy as np
import random
import pdb
import signal
from scipy.stats import beta

#SIGINT handler
def sigint_handler(signal, frame):
    #Do something while breaking
    pdb.set_trace()
    sys.exit(0)

def main():
    instance=sys.argv[2]
    algorithm=sys.argv[4]
    random_seed=int(sys.argv[6])
    epsilon=float(sys.argv[8])
    horizon=int(sys.argv[10])

    np.random.seed(random_seed)
    beta.random_state=random_seed
    f=open(instance,"r")
    f1=f.readlines()
    f.close()
    arms=[]
    for i in f1:
        arms.append(float(i.strip()))
    arms_len=len(arms)
    value=np.zeros(arms_len)
    step_size=np.zeros(arms_len)
    max_reward=max(arms)*horizon
    cum_reward=0
    num_of_success=np.zeros(arms_len)
    num_of_fails=np.zeros(arms_len)
    theta=np.zeros(arms_len)    
    for time_instance in range(horizon):
        for i in range(arms_len):
            theta[i]=beta.rvs(num_of_success[i]+1.0, num_of_fails[i]+1.0)
        action=np.argmax(theta)
        # pdb.set_trace()
        if(np.random.uniform(0,1)<arms[action]):
            # updating step size and the value of the action, reward is 1
            cum_reward+=1
            num_of_success[action]+=1
        else:
            # reward is 0
            num_of_fails[action]+=1
    regret=round(max_reward-cum_reward,2)
    ret_str=instance+", "+algorithm +", "+ str(random_seed)+", "+str(epsilon)+", "+str(horizon)+", "\
    +str(regret)+"\n"
    save_file=open("outputData2.txt","a")
    save_file.write(ret_str)
    save_file.close()

if __name__ == '__main__':
    main()

