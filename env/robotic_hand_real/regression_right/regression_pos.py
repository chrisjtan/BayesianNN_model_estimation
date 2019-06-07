import numpy as np
import pickle
from common.BNN import BNN

with open('../../../data/robotic_hand_real/t42_cyl45_right_data_discrete_v0_d4_m1.obj', 'rb') as pickle_file:
    D, state_dim, action_dim, _, _ = pickle.load(pickle_file, encoding='latin1')

DATA = D

x_data = DATA[:, :10]
y_data = DATA[:, 10:12] - DATA[:, :2]

if __name__ == "__main__":
    neural_network = BNN(nn_type='0')
    neural_network.add_dataset(x_data, y_data, held_out_percentage=0.1)
    neural_network.build_neural_net()
    save_path = '../../../save_model/robotic_hand_real_s4_a6/pos'
    neural_network.train(save_path=save_path, normalization=True, normalization_type='z_score', decay='True')