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

def newton_rhapson(p,q,time_instance,Na):
    f=np.log(time_instance)/Na-p*np.log(p/q)-(1-p)*np.log((1-p)/(1-q))
    f1=(p-q)/(q*(1.0-q))
    h=f/f1
    count=0
    while abs(h)>=1e-9 and count<100:
    # while count<100:
        pdb.set_trace()
        f=np.log(time_instance)/Na-p*np.log(p/q)-(1-p)*np.log((1-p)/(1-q))
        f1=-(q-p)/(q*(1.0-q))
        h=f/f1
        q=min(1-1e-7, max(q-h,p+1e-7))
        count+=1
        # print(h)

    # print(count)
    return q

def main():
    instance=sys.argv[2]
    algorithm=sys.argv[4]
    random_seed=int(sys.argv[6])
    epsilon=float(sys.argv[8])
    horizon=int(sys.argv[10])
    if(instance=="../instances/i-3.txt"):
        pdb.set_trace()
    np.random.seed(random_seed)
    f=open(instance,"r")
    f1=f.readlines()
    f.close()
    arms=[]
    for i in f1:
        arms.append(float(i.strip()))
    arms_len=len(arms)
    step_size=np.zeros(arms_len)
    arms_reward=np.zeros(arms_len)
    max_reward=max(arms)*horizon
    cum_reward=0
    q=np.zeros(arms_len)
    for time_instance in range(horizon):
        # action=0
        if(time_instance<arms_len):
            #initialization action, trying all the arms initiall
            action=time_instance
        else:
            # uncertainity +sample mean
            for i in range(arms_len):
                p = max(arms_reward[i]/step_size[i],1e-7)
                # print(p)
                if(p==1):
                    # action=i
                    q[i]=1
                    break
                q[i] = newton_rhapson(p,p+1e-7, time_instance, step_size[i])  
            action=np.argmax(q)
        # reward for the action
        if(np.random.uniform(0,1)<arms[action]):
            # updating step size and the value of the action, reward is 1
            step_size[action]+=1
            cum_reward+=1
            arms_reward[action]+=1
        else:
            # arms_reward is 0
            step_size[action]+=1
    regret=round(max_reward-cum_reward,2)
    ret_str=instance+", "+algorithm +", "+ str(random_seed)+", "+str(epsilon)+", "+str(horizon)+", "\
    +str(regret)+"\n"
    save_file=open("outputData.txt","a")
    save_file.write(ret_str)
    save_file.close()

if __name__ == '__main__':
    main()

