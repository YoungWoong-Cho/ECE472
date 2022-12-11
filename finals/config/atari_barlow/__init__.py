import torch

from core.config import BaseConfig
from core.dataset import Transforms
from .env_wrapper import AtariBarlowWrapper
from .model import EfficientZeroNet


class AtariBarlowConfig(BaseConfig):
    def __init__(self):
        super(AtariBarlowConfig, self).__init__(
            training_steps=5000,
            last_steps=0,
            test_interval=100,
            log_interval=10,
            vis_interval=1000,
            test_episodes=32,
            checkpoint_interval=100,
            target_model_interval=200,
            save_ckpt_interval=1000,
            max_moves=42,
            test_max_moves=42,
            history_length=400,
            discount=1.0,
            dirichlet_alpha=0.3,
            value_delta_max=0.01,
            num_simulations=200,
            batch_size=64,
            td_steps=42,
            num_actors=1,
            # network initialization/ & normalization
            episode_life=True,
            init_zero=True,
            clip_reward=True,
            # storage efficient
            cvt_string=False,
            image_based=False,
            # lr scheduler
            lr_warm_up=0.01,
            lr_init=0.005,
            lr_decay_rate=1,
            lr_decay_steps=10000,
            auto_td_steps_ratio=0.3,
            # replay window
            start_transitions=8,
            total_transitions=100 * 1000,
            transition_num=1,
            # frame skip & stack observation
            frame_skip=1,
            stacked_observations=1,
            # coefficient
            reward_loss_coeff=1,
            value_loss_coeff=0.25,
            policy_loss_coeff=1,
            consistency_coeff=0.0,
            do_consistency=False,
            # reward sum
            lstm_hidden_size=64,
            lstm_horizon_len=5,
            # siamese
            proj_hid=64,
            proj_out=64,
            pred_hid=32,
            pred_out=64,)
        self.discount **= self.frame_skip
        self.max_moves //= self.frame_skip
        self.test_max_moves //= self.frame_skip

        self.start_transitions = self.start_transitions * 10 // self.frame_skip
        self.start_transitions = max(1, self.start_transitions)

        self.bn_mt = 0.1
        self.blocks = 1  # Number of blocks in the ResNet
        self.channels = 64  # Number of channels in the ResNet
        if self.gray_scale:
            self.channels = 32
        self.reduced_channels_reward = 16  # x36 Number of channels in reward head
        self.reduced_channels_value = 16  # x36 Number of channels in value head
        self.reduced_channels_policy = 16  # x36 Number of channels in policy head
        self.resnet_fc_reward_layers = [32]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [32]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [32]  # Define the hidden layers in the policy head of the prediction network
        self.downsample = True  # Downsample observations before representation network (See paper appendix Network Architecture)
        self.barlow_loss = True
        self.lars_weight_decay = 1.5*1e-6
        self.lars_learning_rate_weights = 0.2
        self.lars_learning_rate_biases = 0.0048

    def visit_softmax_temperature_fn(self, num_moves, trained_steps):
        return 1.0

    def set_game(self, env_name, save_video=False, save_path=None, video_callable=None):
        obs_shape = (3, 6, 7)
        self.obs_shape = (obs_shape[0] * self.stacked_observations, obs_shape[1], obs_shape[2])

        game = self.new_game()
        self.action_space_size = game.action_space_size

    def get_uniform_network(self):
        return EfficientZeroNet(
            self.obs_shape,
            self.action_space_size,
            self.blocks,
            self.channels,
            self.reduced_channels_reward,
            self.reduced_channels_value,
            self.reduced_channels_policy,
            self.resnet_fc_reward_layers,
            self.resnet_fc_value_layers,
            self.resnet_fc_policy_layers,
            self.reward_support.size,
            self.value_support.size,
            self.downsample,
            self.inverse_value_transform,
            self.inverse_reward_transform,
            self.lstm_hidden_size,
            bn_mt=self.bn_mt,
            proj_hid=self.proj_hid,
            proj_out=self.proj_out,
            pred_hid=self.pred_hid,
            pred_out=self.pred_out,
            init_zero=self.init_zero,
            state_norm=self.state_norm)

    def new_game(self, seed=None, save_video=False, save_path=None, video_callable=None, uid=None, test=False, final_test=False):
        return AtariBarlowWrapper(discount=self.discount, cvt_string=self.cvt_string)

    def scalar_reward_loss(self, prediction, target):
        return -(torch.log_softmax(prediction, dim=1) * target).sum(1)

    def scalar_value_loss(self, prediction, target):
        return -(torch.log_softmax(prediction, dim=1) * target).sum(1)

    def set_transforms(self):
        if self.use_augmentation:
            self.transforms = Transforms(self.augmentation, image_shape=(self.obs_shape[1], self.obs_shape[2]))

    def transform(self, images):
        return self.transforms.transform(images)


game_config = AtariBarlowConfig()
