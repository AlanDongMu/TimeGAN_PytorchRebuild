import numpy as np
from sklearn.metrics import accuracy_score
from utils import train_test_divide, extract_time, batch_generator
import torch
import torch.nn as nn


class Discriminator(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layer):
        super(Discriminator, self).__init__()
        self.rnn = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layer, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        d_outputs, d_last_states = self.rnn(X)
        y_hat_logit = self.fc(d_last_states)
        y_hat = self.sigmoid(y_hat_logit)
        return y_hat_logit, y_hat


def discriminative_score_metrics(ori_data, generated_data):
    # Basic Parameters
    no, seq_len, dim = np.asarray(ori_data).shape
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set maximum sequence length and each sequence length
    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(generated_data)
    max_seq_len = max([ori_max_seq_len, generated_max_seq_len])

    train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat = \
        train_test_divide(ori_data, generated_data, ori_time, generated_time)

    # ------ Build a post-hoc RNN discriminator network
    # Network-parameters
    hidden_dim = int(dim / 2)
    iterations = 2000
    batch_size = 128

    discriminator = Discriminator(dim, hidden_dim, 1).to(device)
    optim_discriminator = torch.optim.Adam(discriminator.parameters())
    loss_function = nn.BCEWithLogitsLoss()

    # Training step
    for itt in range(iterations):
        discriminator.train()
        optim_discriminator.zero_grad()
        # Batch setting
        X_mb, T_mb = batch_generator(train_x, train_t, batch_size)
        X_hat_mb, T_hat_mb = batch_generator(train_x_hat, train_t_hat, batch_size)
        # Forward
        X_mb = torch.tensor(X_mb, dtype=torch.float32).to(device)
        X_hat_mb = torch.tensor(X_hat_mb, dtype=torch.float32).to(device)
        y_logit_real, y_pred_real = discriminator(X_mb)
        y_logit_fake, y_pred_fake = discriminator(X_hat_mb)
        # Loss function
        d_loss_real = torch.mean(loss_function(y_logit_real, torch.ones_like(y_logit_real)))
        d_loss_fake = torch.mean(loss_function(y_logit_fake, torch.zeros_like(y_logit_fake)))
        d_loss = d_loss_real + d_loss_fake

        d_loss.backward()
        optim_discriminator.step()

    # ------ Test the performance on the testing set
    test_x = torch.tensor(test_x, dtype=torch.float32).to(device)
    _, y_pred_real_curr = discriminator(test_x)
    y_pred_real_curr = y_pred_real_curr.cpu().detach().numpy()[0]

    test_x_hat = torch.tensor(test_x_hat, dtype=torch.float32).to(device)
    _, y_pred_fake_curr = discriminator(test_x_hat)
    y_pred_fake_curr = y_pred_fake_curr.cpu().detach().numpy()[0]
    y_pred_final = np.squeeze(np.concatenate((y_pred_real_curr, y_pred_fake_curr), axis=0))
    y_label_final = np.concatenate((np.ones([len(y_pred_real_curr), ]), np.zeros([len(y_pred_fake_curr), ])),
                                   axis=0)
    # Compute the accuracy
    acc = accuracy_score(y_label_final, (y_pred_final > 0.5))
    discriminative_score = np.abs(0.5 - acc)
    print('discriminative_score: ', discriminative_score)

    return discriminative_score
