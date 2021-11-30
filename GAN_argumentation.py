import os, sys
sys.path.append(os.getcwd())
import random
import numpy as np
import sklearn.datasets
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import decomposition
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
torch.manual_seed(1)
phosphatase_data = "data/phosphatase.p"
max_seq = 700
GDIM = 512
DDIM = 512
FIXED_GENERATOR = False
LAMBDA = .1
CRITIC_ITERS = 5
ITERS = 80000
use_cuda = True
BATCH_SIZE = 128
class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        main = nn.Sequential(
            nn.Linear(max_seq, GDIM),
            nn.LeakyReLU(0.1),
            nn.Linear(GDIM, GDIM),
            nn.LeakyReLU(0.1),
            nn.Linear(GDIM, GDIM),
            nn.LeakyReLU(0.1),
            nn.Dropout(p = 0.5),
            nn.Linear(GDIM, max_seq),
        )
        self.main = main
        self.lin = nn.Linear(1,21)
        self.softmax = nn.Softmax(-1)

    def forward(self, noise, real_data):
        if FIXED_GENERATOR:
            return real_data
        else:
            output = self.main(noise)
            output = output.unsqueeze(-1)
            output = self.softmax(self.lin(output))
            output = torch.argmax(output,dim=-1)
            return output


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.fc1 = nn.Linear(max_seq, DDIM)
        self.relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(DDIM, DDIM)
        self.relu = nn.LeakyReLU()
        self.fc3 = nn.Linear(DDIM, DDIM)
        self.relu = nn.LeakyReLU()
        self.fc4 = nn.Linear(DDIM, 1)

    def forward(self, inputs):
        out = self.fc1(inputs)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)

        hidden1 = self.relu(self.fc1(inputs))
        hidden2 = self.relu(self.fc2(self.relu(self.fc1(inputs))))
        hidden3 = self.relu(self.fc3(self.relu(self.fc2(self.relu(self.fc1(inputs))))))

        return out.view(-1), hidden1, hidden2, hidden3

def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda() if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates, hidden_output_1, hidden_output_2, hidden_output_3 = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

one = torch.FloatTensor([1])
mone = one * -1
if use_cuda:
    one = one.cuda()
    mone = mone.cuda()
loo = LeaveOneOut()
# GETTING THE TRAINING DATA
with open(phosphatase_data, 'rb') as real:
    real_data = pickle.load(real)
label = []
for rowIndex in range(len(real_data)):
    label.append(1)
# labeling synthesis data
for rowIndex in range(len(data)):
    label.append(0)
labelArray = np.array(label)

    # SET MODEL
netG = Generator()
netD = Discriminator()
if use_cuda:
    netD = netD.cuda()
    netG = netG.cuda()
real_data = torch.from_numpy(real_data).float()
if use_cuda:
    real_data = real_data.cuda()

optimizerD = optim.Adam(netD.parameters(), lr=1e-4)
optimizerG = optim.Adam(netG.parameters(), lr=1e-4)

    # TRAIN
for iteration in range(ITERS):

    for p in netD.parameters():
        p.requires_grad = True

    # TRAIN DISCRIMINATOR
    for iter_d in range(CRITIC_ITERS):

        netD.zero_grad()

        D_real, hidden_output_real_1, hidden_output_real_2, hidden_output_real_3 = netD(real_data)
        D_real = D_real.mean().view(1)
        D_real.backward(mone)

        noise = torch.randn(BATCH_SIZE, max_seq)
        if use_cuda:
            noise = noise.cuda()
        fake = netG(noise, real_data)

        inputv = fake
        D_fake, hidden_output_fake_1, hidden_output_fake_2, hidden_output_fake_3 = netD(inputv)
        D_fake = D_fake.mean().view(1)
        D_fake.backward(one)

        gradient_penalty = calc_gradient_penalty(netD, real_data, fake)
        gradient_penalty.backward()

        D_cost = D_fake - D_real + gradient_penalty
        Wasserstein_D = D_real - D_fake
        optimizerD.step()

    # TRAIN GENERATOR
    if not FIXED_GENERATOR:

        for p in netD.parameters():
            p.requires_grad = False
        netG.zero_grad()

        noise = torch.randn(BATCH_SIZE, max_seq)
        if use_cuda:
            noise = noise.cuda()
        fake = netG(noise, real_data)
        G, hidden_output_ignore_1, hidden_output_ignore_2, hidden_output_ignore_3 = netD(fake)
        G = G.mean().view(1)
        G.backward(mone)
        G_cost = -G
        optimizerG.step()

torch.save(netG.state_dict(), "generator.pt")
prediction_list = []
real_list = []
realFakeFeatures = np.vstack((real_data.data, fake.data))
loo.get_n_splits(realFakeFeatures)
for train_index, test_index in loo.split(realFakeFeatures):
    X_train, X_test = realFakeFeatures[train_index], realFakeFeatures[test_index]
    y_train, y_test = labelArray[train_index], labelArray[test_index]
    knn = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train)
    predicted_y = knn.predict(X_test)
    prediction_list.append(predicted_y)
    real_list.append(y_test)
accuracy = accuracy_score(real_list, prediction_list)
diff_accuracy_05 = abs(accuracy - 0.5)
print(diff_accuracy_05)
    # RECORD BEST EPOCHS
