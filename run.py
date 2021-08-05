import sys

sys.path.append("../code")

from environment.pureva_2D import PurEva_2D_Game
import matplotlib.pyplot as plt

def learn():
    global num_pw
    for eposide in range(EPOSIDES):
        env.set_random_position()
        env.initial_env()
        reward = [0, 0, 0]
        for j in range(env.max_step):
            re, done, results = env.act_and_learn(j)
            for i in range(3):
                reward[i] += re[i]
            if done == True:
                for i in range(3):
                    reward[i] = reward[i]/j
                break
        print("epsidoe",eposide,"reward",reward, results[0], results[1])
        learn_test(eposide)
        # env.plot(show_map = True, show_dis = False, show_reward = False, show_win_rate = False, save_fig = False)
        # plt.show()
    plt.figure('test')
    plt.plot(range(len(num_pw)),num_pw)
    
def learn_test(eposide):
    global num_pw
    if eposide % 100 ==0 and eposide >= 300:
            env.set_random_position()
            test_eopside = 10
            aver_reward = [0,0,0]
            for _ in range(test_eopside):
                env.initial_env()
                reward_test = [0,0,0]
                for step in range(env.max_step):
                    re, done, results = env.act_and_learn(step, eval_f = True)
                    for i in range(3):
                        reward_test[i] += re[i]
                    if done == True:
                        if results[0] == 'purs win':
                            print('purs win')
                            num_pw.append(num_pw[-1] + 1)
                        else:
                            print('evas win')
                            num_pw.append(num_pw[-1] + 0)
                        break
                for i in range(3):
                    aver_reward[i] += reward_test[i]
            for i in range(3):
                aver_reward[i] = aver_reward[i]/test_eopside/env.max_step
            
            print("----------------------------------------")
            print("Test Episodes:",test_eopside," Avg. Reward:",aver_reward)
            print("----------------------------------------")

def final_test():
    final_test_win = 0
    for _ in range(50):
        env.set_random_position()
        env.initial_env()
        for j in range(env.max_step):
            _, done, results = env.act_and_learn(j)
            if done == True:
                if results[0] == 'purs win':
                    final_test_win += 1
                break
    print(final_test_win)

if __name__ == "__main__":
    num_pw = [0]
    EPOSIDES = 10000
    env = PurEva_2D_Game()
    learn()
    env.save_model_trained()
    final_test()
    env.plot()
    plt.show()
