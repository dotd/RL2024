import argparse
import logging
import math
from collections import namedtuple, deque
import time
from pathlib import Path
import os
from PIL import Image
import torch
import random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from itertools import count

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class DQN(nn.Module):

    def __init__(self, h, w, outputs, args, nn_inputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(nn_inputs, args.HIDDEN_LAYER_1, kernel_size=args.KERNEL_SIZE, stride=args.STRIDE)
        self.bn1 = nn.BatchNorm2d(args.HIDDEN_LAYER_1)
        self.conv2 = nn.Conv2d(args.HIDDEN_LAYER_1, args.HIDDEN_LAYER_2, kernel_size=args.KERNEL_SIZE,
                               stride=args.STRIDE)
        self.bn2 = nn.BatchNorm2d(args.HIDDEN_LAYER_2)
        self.conv3 = nn.Conv2d(args.HIDDEN_LAYER_2, args.HIDDEN_LAYER_3, kernel_size=args.KERNEL_SIZE,
                               stride=args.STRIDE)
        self.bn3 = nn.BatchNorm2d(args.HIDDEN_LAYER_3)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        nn.Dropout()
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)  # if we haven't reached full capacity, we append a new transition
        self.memory[self.position] = Transition(*args)
        self.position = (
                                self.position + 1) % self.capacity  # e.g if the capacity is 100, and our position is now 101, we don't append to
        # position 101 (impossible), but to position 1 (its remainder), overwriting old data

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Trainer:

    def __init__(self, args):
        self.EPS_END = args.EPS_END
        self.EPS_START = args.EPS_START
        self.EPS_DECAY = args.EPS_DECAY
        self.N_EPISODES = args.N_EPISODES
        self.FRAMES = args.FRAMES
        self.END_SCORE = args.END_SCORE
        self.BATCH_SIZE = args.BATCH_SIZE
        self.GAMMA = args.GAMMA
        self.TARGET_UPDATE = args.TARGET_UPDATE
        self.MEMORY_SIZE = args.MEMORY_SIZE
        self.LAST_EPISODES_NUM = args.LAST_EPISODES_NUM
        self.TRAINING_STOP = args.TRAINING_STOP

        self.save_graph_folder = 'save_graph/'
        os.makedirs(self.save_graph_folder) if not os.path.exists(self.save_graph_folder) else None
        self.mean_last = deque([0] * self.LAST_EPISODES_NUM, self.LAST_EPISODES_NUM)
        self.graph_name = f'Cartpole_Vision_Stop-{args.TRAINING_STOP}_LastEpNum-{args.LAST_EPISODES_NUM}'

        # Settings for GRAYSCALE / RGB
        if args.GRAYSCALE == 0:
            self.resize = T.Compose([T.ToPILImage(),
                                     T.Resize(args.RESIZE_PIXELS, interpolation=Image.CUBIC),
                                     T.ToTensor()])
            self.nn_inputs = 3 * args.FRAMES  # number of channels for the nn
        else:
            self.resize = T.Compose([T.ToPILImage(),
                                     T.Resize(args.RESIZE_PIXELS, interpolation=Image.BICUBIC),
                                     T.Grayscale(),
                                     T.ToTensor()])
            self.nn_inputs = args.FRAMES  # number of channels for the nn

        self.stop_training = False
        self.env = gym.make('CartPole-v0', render_mode='rgb_array').unwrapped
        self.env.reset()
        plt.figure()

        # Set up matplotlib
        plt.ion()
        self.device = torch.device("cuda" if (torch.cuda.is_available() and args.USE_CUDA) else "cpu")
        if args.GRAYSCALE == 0:
            plt.imshow(self.get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
                       interpolation='none')
        else:
            plt.imshow(self.get_screen().cpu().squeeze(0).permute(
                1, 2, 0).numpy().squeeze(), cmap='gray')
        plt.title('Example extracted screen')
        plt.show()

        eps_threshold = 0.9  # original = 0.9

        init_screen = self.get_screen()
        _, _, screen_height, screen_width = init_screen.shape
        logging.info("Screen height: ", screen_height, " | Width: ", screen_width)

        # Get number of actions from gym action space
        self.n_actions = self.env.action_space.n

        self.policy_net = DQN(screen_height, screen_width, self.n_actions, args, self.nn_inputs).to(self.device)
        self.target_net = DQN(screen_height, screen_width, self.n_actions, args, self.nn_inputs).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        if args.LOAD_MODEL == True:
            policy_net_checkpoint = torch.load('save_model/policy_net_best3.pt')  # best 3 is the default best
            target_net_checkpoint = torch.load('save_model/target_net_best3.pt')
            self.policy_net.load_state_dict(policy_net_checkpoint)
            self.target_net.load_state_dict(target_net_checkpoint)
            self.policy_net.eval()
            self.target_net.eval()
            self.stop_training = True  # if we want to load, then we don't train the network anymore

        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(args.MEMORY_SIZE)

        self.steps_done = 0
        self.episode_durations = []

        for i_episode in range(self.N_EPISODES):
            # Initialize the environment and state
            self.env.reset()
            init_screen = self.get_screen()
            screens = deque([init_screen] * self.FRAMES, self.FRAMES)
            state = torch.cat(list(screens), dim=1)

            for t in count():

                # Select and perform an action
                action = self.select_action(state, self.stop_training)
                state_variables, _, done1, done2, _ = self.env.step(action.item())
                done = done1 or done2

                # Observe new state
                screens.append(self.get_screen())
                next_state = torch.cat(list(screens), dim=1) if not done else None

                # Reward modification for better stability
                x, x_dot, theta, theta_dot = state_variables
                r1 = (self.env.x_threshold - abs(x)) / self.env.x_threshold - 0.8
                r2 = (self.env.theta_threshold_radians - abs(theta)) / self.env.theta_threshold_radians - 0.5
                reward = r1 + r2
                reward = torch.tensor([reward], device=self.device)
                if t >= self.END_SCORE - 1:
                    reward = reward + 20
                    done = 1
                else:
                    if done:
                        reward = reward - 20

                        # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the target network)
                if done:
                    # check_button()  # We check the GUI for inputs
                    self.episode_durations.append(t + 1)
                    self.plot_durations(t + 1)
                    self.mean_last.append(t + 1)
                    mean = 0
                    for i in range(self.LAST_EPISODES_NUM):
                        mean = self.mean_last[i] + mean
                    mean = mean / self.LAST_EPISODES_NUM
                    if mean < self.TRAINING_STOP and self.stop_training == False:
                        self.optimize_model()
                    else:
                        stop_training = 1
                    break

            # Update the target network, copying all weights and biases in DQN
            if i_episode % self.TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

        logging.info('Complete')
        self.env.render()
        self.env.close()
        plt.ioff()
        plt.show()

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        # torch.cat concatenates tensor sequence
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        plt.figure(2)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.mean_last = deque([0] * self.LAST_EPISODES_NUM, self.LAST_EPISODES_NUM)

    def plot_durations(self, score):
        plt.figure(2)
        plt.clf()
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        episode_number = len(durations_t)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy(), label='Score')
        matplotlib.pyplot.hlines(195, 0, episode_number, colors='red', linestyles=':', label='Win Threshold')
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            last100_mean = means[episode_number - 100].item()
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy(), label='Last 100 mean')
            logging.info('Episode: ', episode_number, ' | Score: ', score, '| Last 100 mean = ', last100_mean)
        plt.legend(loc='upper left')
        # plt.savefig('./save_graph/cartpole_dqn_vision_test.png') # for saving graph with latest 100 mean
        plt.pause(0.001)  # pause a bit so that plots are updated

        plt.savefig(self.save_graph_folder + self.graph_name)

    def select_action(self, state, stop_training):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold or stop_training:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)

    def get_cart_location(self, screen_width):
        world_width = self.env.x_threshold * 2
        scale = screen_width / world_width
        return int(self.env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

    # Cropping, downsampling (and Grayscaling) image
    def get_screen(self):
        # Returned screen requested by gym is 400x600x3, but is sometimes larger
        # such as 800x1200x3. Transpose it into torch order (CHW).
        screen = self.env.render().transpose((2, 0, 1))
        # Cart is in the lower half, so strip off the top and bottom of the screen
        _, screen_height, screen_width = screen.shape
        screen = screen[:, int(screen_height * 0.4):int(screen_height * 0.8)]
        view_width = int(screen_width * 0.6)
        cart_location = self.get_cart_location(screen_width)
        if cart_location < view_width // 2:
            slice_range = slice(view_width)
        elif cart_location > (screen_width - view_width // 2):
            slice_range = slice(-view_width, None)
        else:
            slice_range = slice(cart_location - view_width // 2,
                                cart_location + view_width // 2)
        # Strip off the edges, so that we have a square image centered on a cart
        screen = screen[:, :, slice_range]
        # Convert to float, rescale, convert to torch tensor
        # (this doesn't require a copy)
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        # Resize, and add a batch dimension (BCHW)
        return self.resize(screen).unsqueeze(0).to(self.device)

    def reset(self):
        self.env.reset()
        plt.figure()


def run_main(args):
    trainer = Trainer(args)


def prepare_parameters_and_logging():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        default="DEBUG", help="Set the logging level (default is DEBUG).")
    parser.add_argument("--BATCH_SIZE", type=int, default=128)
    parser.add_argument("--GAMMA", type=float, default=0.999)
    parser.add_argument("--EPS_START", type=float, default=0.9)
    parser.add_argument("--EPS_END", type=float, default=0.05)
    parser.add_argument("--EPS_DECAY", type=int, default=5000)
    parser.add_argument("--TARGET_UPDATE", type=int, default=50)
    parser.add_argument("--MEMORY_SIZE", type=int, default=100000)
    parser.add_argument("--END_SCORE", type=int, default=200)
    parser.add_argument("--USE_CUDA", type=bool, default=False)
    parser.add_argument("--LOAD_MODEL", type=bool, default=False)
    parser.add_argument("--GRAYSCALE", type=bool, default=True)
    parser.add_argument("--TRAINING_STOP", type=int, default=142)
    parser.add_argument("--N_EPISODES", type=int, default=50000)
    parser.add_argument("--LAST_EPISODES_NUM", type=int, default=20)
    parser.add_argument("--FRAMES", type=int, default=2)
    parser.add_argument("--RESIZE_PIXELS", type=int, default=60)
    parser.add_argument("--HIDDEN_LAYER_1", type=int, default=16)
    parser.add_argument("--HIDDEN_LAYER_2", type=int, default=32)
    parser.add_argument("--HIDDEN_LAYER_3", type=int, default=32)
    parser.add_argument("--KERNEL_SIZE", type=int, default=5)
    parser.add_argument("--STRIDE", type=int, default=2)
    parser.add_argument("--signature", type=str, default=time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--model_name", type=str, default='breakout_model')
    parser.add_argument("--model_save_interval", type=int, default=11)
    parser.add_argument("--logging_interval", type=int, default=4)
    parser.add_argument("--FONT", type=str, default="Fixedsys 12 bold")

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
    run_main(args)


if __name__ == '__main__':
    main()
    # cProfile.run('main()')  # Run the main function with cProfile
