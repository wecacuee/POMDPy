import logging
import json

from collections import namedtuple

import cv2
from generative_inpainting.test import FillInpainting

from pomdpy.pomdp import Model, StepResult
from pomdpy.util import console, config_parser


State = namedtuple('State', ['map', 'pose', 'targets'])
Belief = namedtuple('Belief', ['map', 'pose', 'targets'])


class SeekingModel(Model):
    def __init__(self, args):
        super().__init__(args)
        self.logger = logging.getLogger('POMDPy.SeekingModel')
        self.config = json.load(open(config_parser.seeking_cfg, "r"))
        self.prediction_model = FillInpainting(
            self.config['checkpoint_dir'],
            self.config['prediction_model_config_yaml'])
        self.map_size = self.config['map_size']
        self.pose_size = self.config['pose_size']
        self.initialize()

    def initialize(self):
        pass


    def reset_for_simulation(self):
        """
        The Simulator (Model) should be reset before each simulation
        :return:
        """

    def reset_for_epoch(self):
        """
        Defines behavior for resetting the simulator before each epoch
        :return:
        """

    def update(self, sim_data):
        """
        Update the state of the simulator with sim_data
        :param sim_data:
        :return:
        """
        raise NotImplementedError()

    def _predict_image_from_unknown(self, image, unknown):
        erode_frac = erode_frac_min + np.random.rand() * (
            erode_frac_max - erode_frac_min)
        Dx, Dy = image.shape[:2:-1]
        kernel = np.ones((int(erode_frac * Dx),int(erode_frac * Dy)),np.uint8)
        expand_unknown = cv2.erode(unknown.astype(np.uint8), kernel)
        diff_unknown = unknown & (~expand_unknown)
        next_map_image = self.prediction_model.predict(image, diff_unknown)
        return next_map_image

    def generate_step(self, state, action):
        """
        Generates a full StepResult, including the next state, an observation, and the reward
        *
        * For convenience, the action taken is also included in the result, as well as a flag for
        * whether or not the resulting next state is terminal.
        :param state: [ OccupancyGrid with only {0, 1}?, Pose of the robot]
        :param action: [ Velocity vector ] s.t. Pose + Velocity = Next pose
        :return: StepResult
        """
        # Separate the occupancygrid into known and unknown area with a mask
        occgrid = state.map
        unknown = occgrid == -1
        image = occgrid.copy()
        image[unknown] = occgrid.max() / 2
        image = np.uint8(image * 255 / image.max())

        next_map_image = self._predict_image_from_unknown(image, unknown)
        unknown_area = ((next_map_image < 200) | (next_map_image > 50))
        next_map_image[unknown_area] = -1

        nest_state = State()
        next_state.map = next_map_image

        # Predict the next action
        velocity = action
        next_state.pose = state.pose + velocity

        return StepResult(next_state=next_state, action=action)


    def sample_an_init_state(self):
        """
        Samples an initial state from the initial belief.
        :return: State
        """
        return State(map=np.random.randint(-1, 2, size=self.map_size),
                     pose=np.random.rand(size=self.pose_size))

    def sample_state_uninformed(self):
        """
        Samples a state from a poorly-informed prior. This is used by the provided default
        implementation of the second generateParticles() method.
        :return:
        """
        return self.sample_an_init_state()

    def sample_state_informed(self, belief):
        """
        :param belief:
        :return:
        """
        map_belief = belief.map # [0, 1] \in R^{map_size}
        pose_belief = belief.pose
        threshold = np.random.rand(size=self.map_size) * 0.9
        map_ = (map_belief > threshold + 0.05)
        return State(map=map_,
                     pose=np.random.multivariate_normal(
                         pose_belief.mean, pose_belief.cov))

    def get_initial_belief_state(self):
        """
        Return an np.array of initial belief probabilities for each state
        :return:
        """
        return Belief(map=np.ones(self.map_size) * 0.5,
                      pose=np.zeros(self.pose_size))

    def belief_update(self, old_belief, action, observation):
        """
        Use bayes filter to update belief distribution
        :param old_belief:
        :param action
        :param observation
        :return:
        """
        # map_belief = old_belief.map
        # map_image = np.uint8(map_belief * 255)
        # unknown = ((map_belief < 200) | (map_belief > 50))
        # next_map_image = self._predict_image_from_unknown(image, unknown)
        # next_map_belief = next_map_image / 255.0
        # TODO ROS SLAM
        #return updated_map + updated_pose from gmapping? or omnimapper?


    def get_all_states(self):
        """
        :return: list of enumerated states (discrete) or range of states (continuous)
        """
        min_state = State(map=np.zeros(self.map_size, dtype=np.uint8),
                          pose=self.min_pose)
        max_state = State(map=np.ones(self.map_size, dtype=np.uint8) * 255,
                          pose=self.max_pose)
        return (min_state, max_state)

    def get_transition_matrix(self):
        """
        Transition probability matrix, for value iteration
        :return:
        """
        raise RuntimeError("Connote compute")

    def get_all_actions(self):
        """
        :return: list of enumerated actions (discrete) or range of actions (continuous)
        """
        return (-self.max_velocity, self.max_velocity)

    def get_all_observations(self):
        """
        :return: list of enumerated observations (discrete) or range of observations (continuous)
        """
        return (np.zeros(self.nscans), np.ones(self.nscans) * self.max_laser_range)

    def get_observation_matrix():
        """
        Observation probability matrix
        :return:
        """
        raise RuntimeError("Cannot compute")

    def get_reward_matrix():
        """
        Return reward matrix
        :return:
        """
        raise RuntimeError("Cannot compute")

    def get_legal_actions(self, state):
        """
        Given the current state of the system, return all legal actions
        :return: list of legal actions
        """
        raise NotImplementedError("have to think about it")

    def is_terminal(self, state):
        """
        Returns true iff the given state is terminal.
        :param state:
        :return:
        """
        return False

    def is_valid(self, state):
        """
        Returns true iff the given state is valid
        :param state:
        :return:
        """
        return True

    def create_action_pool(self):
        """
        :param solver:
        :return:
        """
        #raise NotImplementedError()

    def create_root_historical_data(self, solver):
        """
        reset smart data for the root of the belief tree, if smart data is being used
        :return:
        """
        raise NotImplementedError()

    def get_max_undiscounted_return(self):
        """
        Calculate and return the highest possible undiscounted return
        :return:
        """
        raise NotImplementedError()

    def create_observation_pool(self, solver):
        """
        Return a concrete observation pool (discrete or continuous)
        :param solver:
        :return:
        """
        raise NotImplementedError()
