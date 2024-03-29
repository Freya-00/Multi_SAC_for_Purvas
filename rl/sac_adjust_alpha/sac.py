import os
import sys
sys.path.append("../code")
import torch
import torch.nn.functional as F
from torch.optim import Adam
from rl.sac_adjust_alpha.utils import soft_update, hard_update
from rl.sac_adjust_alpha.model import GaussianPolicy, QNetwork, DeterministicPolicy

use_cuda = torch.cuda.is_available()

class SAC(object):
    def __init__(self, 
                num_inputs, 
                action_dim,
                hidden_size = 256,
                learning_rate = 0.0003,
                policy_type = "Gaussian",
                share_action = False,
                flag_automatic_entropy_tuning = False):
        
        self.gamma = 0.98 # discount factor for reward
        self.tau = 0.005 # target smoothing coefficient(τ)
        self.alpha = 0.1 # Temperature parameter α determines the relative importance of the entropy term against the reward
        self.share_action = share_action # Share actions between agents
        self.share_ac_num = 9 # the num of shared actions
        self.policy_type = policy_type # Policy Type: Gaussian | Deterministic
        self.automatic_entropy_tuning = flag_automatic_entropy_tuning # Automaically adjust α

        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.critic = QNetwork(num_inputs, action_dim, hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=learning_rate)

        self.critic_target = QNetwork(num_inputs, action_dim, hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_dim).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=learning_rate)
            
            if self.share_action is True:
                self.policy = GaussianPolicy(num_inputs - self.share_ac_num, action_dim, hidden_size).to(self.device)
            else:
                self.policy = GaussianPolicy(num_inputs, action_dim, hidden_size).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=learning_rate)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            if self.share_action is True:
                self.policy = DeterministicPolicy(num_inputs - self.share_ac_num, action_dim, hidden_size).to(self.device)
            else:
                self.policy = DeterministicPolicy(num_inputs, action_dim, hidden_size).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=learning_rate)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size = 1024):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)
        if self.share_action is True:
            state_of_policy = state_batch[:,:-self.share_ac_num]
            state_of_policy_next = next_state_batch[:,:-self.share_ac_num]
        else:
            state_of_policy = state_batch
            state_of_policy_next = next_state_batch
        
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        state_of_policy = torch.FloatTensor(state_of_policy).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        state_of_policy_next = torch.FloatTensor(state_of_policy_next).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(state_of_policy_next)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_of_policy)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs

        soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')
        if actor_path is None:
            actor_path = "models/sac_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/sac_critic_{}_{}".format(env_name, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    # Load model parameters
    def load_model(self, actor_path , critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path, map_location=torch.device('cpu')))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path, map_location=torch.device('cpu')))

