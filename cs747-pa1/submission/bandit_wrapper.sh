#!/bin/bash
epsilon="0"

# for file_instance in 1.txt 2.txt 3.txt
for file_instance in 3.txt
do
    # for algo in round-robin epsilon-greedy ucb thompson-sampling
    for algo in kl-ucb
    do
        if [[ $algo == 'epsilon-greedy' ]]; then
            for epsilon in 0.002 0.02 0.2
            do
                for horizon in 50 200 800 3200 12800 51200 204800
                do
                    echo "$file_instance running $algo     epsilon:$epsilon    horizon:$horizon"
                    for seed in {0..49}
                    do
                        ./bandit.sh --instance ../instances/i-$file_instance --algorithm $algo --randomSeed "$seed" --epsilon $epsilon --horizon $horizon    
                    done
                done
            done
        else
            for horizon in 50 200 800 3200 12800 51200 204800
            # for horizon in 
            do
                echo "running $file_instance    $algo     epsilon:$epsilon    horizon:$horizon"
                for seed in {0..49}
                # for seed in {27..49}
                do
                    ./bandit.sh --instance ../instances/i-$file_instance --algorithm $algo --randomSeed "$seed" --epsilon $epsilon --horizon $horizon    
                done
            done
        fi
    done
done