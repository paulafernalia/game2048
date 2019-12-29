from utils import ReplayBuffer
from agents import DeepQ_MLP
from environments import Game2048
from matplotlib.animation import FFMpegWriter

# import cProfile

BUFFER_SIZE = 1000000
WARMUP_ACTIONS = 50000
C_TARGET_UPDATE = 10000
MINIBATCH_SIZE = 32
LEARNING_RATE = .0005
DISCOUNT_FACTOR = .99999
EPSILON = .5
DISCOUNT_EPSILON = .9995
BETA_1 = .95
PLOT_FREQ = 50
PRINT_FREQ = 1
MAX_ITER = 20


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

        if (episode % PLOT_FREQ == 0) and (agent.a_count >= agent.warmup):
            print('check')
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
            action, qmax = agent.qlearning_action(phi, env.tabu)
            reward = env.step(action)

            # add this tuple (s0, a, r, d, s1) to the memory
            memory.add(phi, action, reward, env.done, env.encode_state())

            # if within the initial replay buffer start, do nothing
            if agent.a_count >= agent.warmup:

                print('check')
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
            print("* Episode: {} score: {} actions: {}".format(
                  episode, int(env.score), agent.a_count))

        # # close moviewriter
        if plot_this_episode and (agent.a_count >= agent.warmup):
            writer.finish()

        # reset environment 2048
        env.reset()


if __name__ == "__main__":
    main()
    # cProfile.run('main()') 