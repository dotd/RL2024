from collections import deque
from pathlib import Path

import gym
import logging
import argparse
import os

import torch
import numpy as np
import random
import math
import time
from PIL import Image

import torchvision.transforms as transforms

from tst.breakout_baseline.networks import DQN
from tst.breakout_baseline.replay_buffer import ReplayBuffer


# Class to convert images to grayscale and crop
class Transforms:
    def to_gray(frame1, frame2=None):
        gray_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.CenterCrop((175, 150)),
            transforms.Resize((84, 84)),
            transforms.ToTensor()
        ])

        # Subtract one frame from the other to get sense of ball and paddle direction
        if frame2 is not None:
            new_frame = gray_transform(frame2) - 0.4 * gray_transform(frame1)
        else:
            new_frame = gray_transform(frame1)

        return new_frame.numpy()


class DQAgent(object):
    # Take hyperparameters, as well as openai gym environment name
    # Keeps the environment in the class. All learning/playing functions are built in
    def __init__(
            self,
            replace_target_cnt,
            state_space,
            action_space,
            gamma,
            eps_strt,
            eps_end,
            eps_dec,
            batch_size,
            lr,
            model_save_interval,
            model_name,
            signature):

        # Set global variables
        self.state_space = state_space
        self.action_space = action_space
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.eps_strt = eps_strt
        self.eps = self.eps_strt
        self.eps_dec = eps_dec
        self.eps_end = eps_end
        self.model_save_interval = model_save_interval

        self.signature = signature
        logging.info(f"{self.__str__()}")

        # Use GPU if available
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Initialise Replay Memory
        self.memory = ReplayBuffer()

        # After how many training iterations the target network should update
        self.replace_target_cnt = replace_target_cnt
        self.learn_counter = 0

        # Initialise policy and target networks, set target network to eval mode
        self.policy_net = DQN(self.state_space, self.action_space, filename=model_name).to(self.device)
        self.target_net = DQN(self.state_space, self.action_space, filename=model_name + 'target').to(self.device)
        self.target_net.eval()

        # If pretrained model of the modelname already exists, load it
        try:
            self.policy_net.load_model()
            logging.info('loaded pretrained model')
        except:
            logging.info('create new model')

        # Set target net to be the same as policy net
        self.replace_target_net()

        # Set optimizer & loss function
        self.optim = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.loss = torch.nn.SmoothL1Loss()

    def sample_batch(self):
        batch = self.memory.sample_batch(self.batch_size)
        state_shape = batch.state[0].shape

        # Convert to tensors with correct dimensions
        state = torch.tensor(batch.state).view(self.batch_size, -1, state_shape[1], state_shape[2]).float().to(
            self.device)
        action = torch.tensor(batch.action).unsqueeze(1).to(self.device)
        reward = torch.tensor(batch.reward).float().unsqueeze(1).to(self.device)
        state_ = torch.tensor(batch.state_).view(self.batch_size, -1, state_shape[1], state_shape[2]).float().to(
            self.device)
        done = torch.tensor(batch.done).float().unsqueeze(1).to(self.device)

        return state, action, reward, state_, done

    # Returns the greedy action according to the policy net
    def greedy_action(self, obs):
        obs = torch.tensor(obs).float().to(self.device)  # C x H x W
        obs = obs.unsqueeze(0)  # B=1 x C x H x W
        action = self.policy_net(obs).argmax().item()
        return action

    # Returns an action based on epsilon greedy method
    def choose_action(self, obs):
        if random.random() > self.eps:
            action = self.greedy_action(obs)
        else:
            action = random.choice([x for x in range(self.action_space)])
        return action

    # Stores a transition into memory
    def store_transition(self, *args):
        self.memory.add_transition(*args)

    # Updates the target net to have same weights as policy net
    def replace_target_net(self):
        if self.learn_counter % self.replace_target_cnt == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            logging.info('Target network replaced')

    # Decrement epsilon
    def dec_eps(self):
        self.eps = self.eps - self.eps_dec if self.eps > self.eps_end \
            else self.eps_end

    # Samples a single batch according to batchsize and updates the policy net
    def learn(self, num_iters, episode):
        if self.memory.pointer < self.batch_size:
            return

        for i in range(num_iters):
            # Sample batch
            state, action, reward, state_, done = self.sample_batch()

            # Calculate the value of the action taken
            q_eval = self.policy_net(state).gather(1, action)

            # Calculate best next action value from the target net and detach from graph
            q_next = self.target_net(state_).detach().max(1)[0].unsqueeze(1)
            # Using q_next and reward, calculate q_target
            # (1-done) ensures q_target is 0 if transition is in a terminating state
            q_target = (1 - done) * (reward + self.gamma * q_next) + (done * reward)

            # Compute the loss
            # loss = self.loss(q_target, q_eval).to(self.device)
            loss = self.loss(q_eval, q_target).to(self.device)

            # Perform backward propagation and optimization step
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # Increment learn_counter (for dec_eps and replace_target_net)
            self.learn_counter += 1

            # Check replace target net
            self.replace_target_net()

        # Save model & decrement epsilon
        if episode % self.model_save_interval == 0:
            self.policy_net.save_model(signature=self.signature, episode=episode)
        self.dec_eps()

    # Save gif of an episode starting num_transitions ago from memory
    def save_gif(self, num_transitions, episode_num=-1, signature="default_signature"):
        frames = []
        for i in range(self.memory.pointer - num_transitions, self.memory.pointer):
            frame = Image.fromarray(self.memory.buffer[i].raw_state, mode='RGB')
            frames.append(frame)
        # check if folder exists
        os.makedirs(f'./gifs/{signature}') if not os.path.exists(f'./gifs/{signature}') else None
        frames[0].save(f'./gifs/{signature}/episode_{episode_num}.gif', format='GIF', append_images=frames[1:],
                       save_all=True, duration=10, loop=0)

    def __str__(self):
        return (f"state_space={self.state_space}\naction_space={self.action_space}\nbatch_size={self.batch_size}\n"
                f"gamma={self.gamma}\neps_strt={self.eps_strt}\neps_end={self.eps_end}\n"
                f"eps_dec={self.eps_dec}\nrl={self.lr}\n")


# Plays num_eps amount of games, while optimizing the model after each episode


def train(env, agent, args):
    scores = []
    steps_counter = 0

    max_score = 0
    start_time = time.time()
    for episode in range(args.num_episodes):
        done = False

        # Reset environment and preprocess state
        obs, info = env.reset()
        steps_counter += 1
        state = Transforms.to_gray(obs)

        score = 0
        cnt = 0
        q = deque(maxlen=args.frames_per_rl)
        while not done:
            # Take epsilon greedy action
            action = agent.choose_action(state)
            obs_next, reward, done, truncated, _ = env.step(action)
            steps_counter += 1
            # Preprocess next state and store transition
            state_next = Transforms.to_gray(obs, obs_next)
            agent.store_transition(state, action, reward, state_next, int(done), obs)

            score += reward
            state = state_next
            cnt += 1

        # Maintain record of the max score achieved so far
        if score > max_score:
            max_score = score

        # Save a gif if episode is best so far
        if score > 300 and score >= max_score or episode >= 1 and episode % args.save_gif_interval == 1:
            agent.save_gif(cnt, episode_num=episode, signature=args.signature)

        scores.append(score)
        # Train on as many transitions as there have been added in the episode

        agent.learn(num_iters=math.ceil(cnt / agent.batch_size), episode=episode)

        if episode % args.logging_interval == 0 or episode == args.num_episodes - 1:
            logging.info(
                f'[{time.time() - start_time:.2f}][{episode / args.num_episodes * 100:.2f}%]\t'
                f'Episode:{episode:6}/{args.num_episodes} Score:{score}\t'
                f'AvgScore(100):{np.mean(scores[-100:]):.4f}\t' +
                f'Epsilon: {agent.eps:.3}\tTransitionsAdded:{cnt}\tLearning x{math.ceil(cnt / agent.batch_size)}\t' +
                f'MaxScore:{max_score}\tlen(RB):{len(agent.memory)}\t'
                f'mean(episode)[sec]{(time.time() - start_time) / (episode + 1):.4f}')

    env.close()


def run_breakout(args):
    env = gym.make(args.environment)
    obs, info = env.reset()
    state = Transforms.to_gray(obs)
    state_space = state.shape
    action_space = env.action_space.n
    logging.debug(f"state: {state_space}")
    logging.debug(f"action: {action_space}")
    logging.debug(f"info: {info}")
    logging.info(f"Create the agent")
    agent = DQAgent(
        state_space=state_space,
        action_space=action_space,
        model_name=args.model_name,
        replace_target_cnt=args.replace_target_cnt,
        gamma=args.gamma,
        eps_strt=args.eps_start,
        eps_end=args.eps_end,
        eps_dec=args.eps_dec,
        batch_size=args.batch_size,
        lr=args.lr,
        model_save_interval=args.model_save_interval,
        signature=args.signature
    )
    train(env=env,
          agent=agent,
          args=args)
    env.close()


def prepare_parameters_and_logging():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        default="DEBUG", help="Set the logging level (default is DEBUG).")
    parser.add_argument("--render", type=bool, default=False)
    parser.add_argument("--replace_target_cnt", type=int, default=5000)
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--eps_start", type=float, default=0.1)
    parser.add_argument("--eps_end", type=float, default=0.001)
    parser.add_argument("--eps_dec", type=float, default=5e-6)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--save_gif_interval", type=int, default=10)
    parser.add_argument("--signature", type=str, default=time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--model_name", type=str, default='breakout_model')
    parser.add_argument("--model_save_interval", type=int, default=11)
    parser.add_argument("--logging_interval", type=int, default=4)
    parser.add_argument("--frames_per_rl", type=int, default=4)
    parser.add_argument("--environment", choices=["Breakout-v4", "CartPole-v1"], default="Breakout-v4",
                        help="Set environment")

    args = parser.parse_args()
    filename_full = os.path.abspath(__file__)
    filename = filename_full.split("/")[-1].split(".")[0]
    log_file = f"./logs/{filename}_{args.signature}.log"
    log_file_latest = f"./logs/{filename}.log"
    os.makedirs("./logs/") if not os.path.exists("./logs/") else None
    logging.basicConfig(level=getattr(logging, args.log_level), handlers=[logging.FileHandler(log_file),
                                                                          logging.FileHandler(log_file_latest),
                                                                          logging.StreamHandler()])
    # Put into logging the file itself for debugging
    logging.info(f"Running file: {filename_full}")
    logging.info(f"log_file: {log_file}")
    logging.info((f"Log file:\n{'start_of_running_file'.upper()}\n"
                  f"{Path(filename_full).read_text()}\n{'end_of_running_file'.upper()}"))
    for arg, value in vars(args).items():
        logging.info(f"{__file__.split('/')[-1]}> {arg}: {value}")
    return args


def main():
    args = prepare_parameters_and_logging()
    run_breakout(args)


if __name__ == '__main__':
    main()
    # cProfile.run('main()')  # Run the main function with cProfile
