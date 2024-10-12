from maze import Maze
from RL import QLearning
#save the data(logs) into csv file
import csv

def update():
    i = 0
    j = 0
    with open('maze_logs.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Episode', 'Steps', 'Result'])

        for episode in range(1000):
            # initial observation
            observation = env.reset()
            i = i + 1
            print("Episode: ", i)
            k = 0
            while True:
                k = k + 1
                # fresh env
                env.render()

                # RL choose action based on observation
                action = RL.choose_action(str(observation))

                # RL take action and get next observation and reward
                observation_, reward, done = env.step(action)

                # RL learn from this transition
                RL.learn(str(observation), action, reward, str(observation_))

                # swap observation
                observation = observation_

                # break while loop when end of this episode
                if done:
                    print("Steps: ", k)
                    result = "success" if reward == 1 else "fail"
                    if reward == 1:
                        print("success")
                        if k < j or j == 0:
                            j = k
                    else:
                        print("fail")
                    writer.writerow([i, k, result])
                    print(" ")
                    break

    # end of game
    print('minimum steps: ', j)
    #print the final q table
    print(RL.q_table)
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = QLearning(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()
