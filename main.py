from utils import ReplayBuffer
from agents import DeepQ_MLP
from environments import Game2048
from matplotlib.animation import FFMpegWriter
import pdb
import pickle

# import cProfile

BUFFER_SIZE = 500000
WARMUP_ACTIONS = 2  # 25000  # 50000
C_TARGET_UPDATE = 5000
MINIBATCH_SIZE = 32
LEARNING_RATE = .005
DISCOUNT_FACTOR = .9999
EPSILON = .1
DISCOUNT_EPSILON = 1  # .9995
BETA_1 = .95
PLOT_FREQ = 1000
PRINT_FREQ = 100
MAX_ITER = 10000


def main():

    # create game, agent and replaybuffer instances
    agent = DeepQ_MLP(input_shape=(16*13,),
                      epsilon=EPSILON,
                      discount_factor=DISCOUNT_FACTOR,
                      learning_rate=LEARNING_RATE,
                      beta_1=BETA_1,
                      warmup=WARMUP_ACTIONS,
                      discount_epsilon=DISCOUNT_EPSILON,
                      c=C_TARGET_UPDATE)

    memory = ReplayBuffer(BUFFER_SIZE)
    env = Game2048()
    env.reset()
    progress = {'scores': [], 'residuals': []}

    metadata = dict(title='GAME2048', artist='paulafernalia')
    writer = FFMpegWriter(fps=4, metadata=metadata)

    # Run a number of episodes
    for episode in range(MAX_ITER):

        # pdb.set_trace()

        if (episode % PLOT_FREQ == 0) and (agent.a_count >= agent.warmup):
            print()
            print('* Plot this episode {}'.format(episode))
            plot_this_episode = True
            env.render(iter_=episode)
            writer.setup(fig=env.fig, dpi=300,
                         outfile='episode{}.mp4'.format(episode))
        else:
            plot_this_episode = False

        while not(env.done):

            # store previous state
            phi = env.encode_state()

            # take one action -> next step and reward
            # this action will be random with probability epsilon
            # and always random before warmup
            action, qmax = agent.eps_greedy_action(phi, env.tabu)
            reward = env.step(action)

            # add this tuple (s0, a, r, d, s1) to the memory
            memory.add(phi, action, reward, env.done, env.encode_state())

            # if within the initial replay buffer start, do nothing
            if agent.a_count >= agent.warmup:

                pdb.set_trace()

                # sample random minibatch of transitions
                minibatch = memory.sample(MINIBATCH_SIZE)

                # update plot
                if plot_this_episode:
                    env.render(wait=.0, iter_=episode)
                    writer.grab_frame()

                # perform one step of stochastic gradient descent
                agent.one_step_gd(minibatch)

                # add value to convergence
                action_i = agent.action_str2idx(action)
                new_qmax = agent.Qmodel.predict(phi, batch_size=1)[0, action_i]
                progress['residuals'].append(abs(qmax - new_qmax))

        # add value to scores
        progress['scores'].append(env.score)

        # # print progress
        if (episode % PRINT_FREQ == 0):
            print()
            print("* Episode: {} score: {} actions: {}".format(
                  episode, int(env.score), agent.a_count))
            # print(agent.c_count)
            print('.   - main model  ', agent.Qmodel.get_weights()[0][0, 0])
            print('.   - target model', agent.target_Qmodel.get_weights()[0][0, 0])

        # # close moviewriter
        if plot_this_episode and (agent.a_count >= agent.warmup):
            writer.finish()

        # reset environment 2048
        env.reset()

        # pdb.set_trace()
        if env.done:
            pdb.set_trace()

    with open('results.pickle', 'wb') as f:
        pickle.dump(progress, f)


if __name__ == "__main__":
    main()
