import sys

sys.path.append("../code")

from environment.pureva2D_decentral import PurEva_2D_Game
import matplotlib.pyplot as plt

if __name__ == "__main__":
    EPOSIDES = 8000
    env = PurEva_2D_Game()
    for eposide in range(EPOSIDES):
        if eposide >= 300 :
            'random position'
            env.set_random_position()
        'learn'
        env.initial_env()
        reward = [0, 0, 0]
        for j in range(env.max_step):
            re, done = env.act_and_learn(j)
            for i in range(3):
                reward[i] += re[i]
            if done == True:
                for i in range(3):
                    reward[i] = reward[i]/j
                break

            # plt.show()
        print("epsidoe",eposide,"reward",reward)
        'plot'
        # if eposide % 100  == 0 and eposide !=0:
        #     env.plot()
        #     plt.show()
        
        
        'test'
        if eposide % 100 == 0:
            test_eopside = 10
            aver_reward = [0,0,0]
            for _ in range(test_eopside):
                env.initial_env()
                reward_test = [0,0,0]
                for step in range(env.max_step):
                    re, done = env.act_and_learn(step, eval_f = True)
                    for i in range(3):
                        reward_test[i] += re[i]
                    if done == True:
                        break
                for i in range(3):
                    aver_reward[i] += reward_test[i]
                # if eposide > 200:
                #     env.plot(show_map = True, show_dis = False, show_reward = False, show_win_rate = False)
                #     plt.show()
            for i in range(3):
                aver_reward[i] = aver_reward[i]/test_eopside/env.max_step
            
            print("----------------------------------------")
            print("Test Episodes:",test_eopside," Avg. Reward:",aver_reward)
            print("----------------------------------------")
     
    env.plot()
    plt.show()
