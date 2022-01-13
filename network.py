"""Small network to learn bitwise_xor operation."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torch.utils.data import DataLoader
from tqdm import tqdm


# dataset generator
class XorDataset(data.Dataset):
    """Create dataset for Xor learning."""

    def __init__(self, nsample=1000, test=False):
        """Init the dataset
        :returns: TODO

        """
        data.Dataset.__init__(self)
        self.nsample = nsample
        self.test = test
        if test:
            self.input_vars = torch.tensor([[1, 1], [1, 0], [0, 1], [0, 0]],
                                           dtype=torch.float)
            self.nsample = 4
        else:
            self.input_vars = torch.bernoulli(
                torch.ones((self.nsample, 2)) * 0.5)

    def __getitem__(self, index):
        """Get a data point."""
        assert index < self.nsample, "The index must be less than the number of samples."
        inp = self.input_vars[index]
        return inp, torch.logical_xor(*inp).type(torch.float)

    def __len__(self):
        """Return len of the dataset."""
        return self.nsample


class OrDataset(data.Dataset):
    """Create dataset for Or learning."""

    def __init__(self, nsample=1000, test=False):
        """Init the dataset
        :returns: TODO

        """
        data.Dataset.__init__(self)
        self.nsample = nsample
        self.test = test
        if test:
            self.input_vars = torch.tensor([[1, 1], [1, 0], [0, 1], [0, 0]],
                                           dtype=torch.float)
            self.nsample = 4
        else:
            self.input_vars = torch.bernoulli(
                torch.ones((self.nsample, 2)) * 0.5)

    def __getitem__(self, index):
        """Get a data point."""
        assert index < self.nsample, "The index must be less than the number of samples."
        inp = self.input_vars[index]
        return inp, torch.logical_or(*inp).type(torch.float)

    def __len__(self):
        """Return len of the dataset."""
        return self.nsample


# Classifier


class XorNet(nn.Module):
    """A simple network to predict Xor value"""

    def __init__(self, hid_size=6):
        """Initialize the network."""
        nn.Module.__init__(self)

        self.hid_size = hid_size
        self.hid_layer = nn.Linear(in_features=2, out_features=self.hid_size)
        self.out_layer = nn.Linear(in_features=self.hid_size, out_features=1)

    def forward(self, x):
        """Compute epoch into the neural net.

        :x: 2D input
        :returns: The network evaluated at x

        """
        x = self.hid_layer(x)
        x = torch.sigmoid(x)
        x = self.out_layer(x)
        x = torch.sigmoid(x)

        return x.squeeze()


class Trainer:
    """A simple trainer to train a simple network."""

    def __init__(self,
                 net=XorNet(hid_size=6),
                 trainset=XorDataset(nsample=2000),
                 testset=XorDataset(test=True),
                 criterion=nn.MSELoss(),
                 optimizer=optim.Adam,
                 learning_rate=1e-3,
                 error_rate=1e-6,
                 num_epochs=300,
                 batch_size=64):
        """Initialize the trainer and print some data.

        :learning_rate: TODO
        :error_rate: TODO
        :num_epochs: TODO
        :batch_size: TODO
        :returns: TODO

        """
        self.trainset = trainset
        self.testset = testset
        self.learning_rate = learning_rate
        self.error_rate = error_rate

        self.num_epochs = num_epochs
        self.num_w = 4
        self.bs = batch_size

        print("Training with :")
        print("Num epochs:", self.num_epochs)
        print("Batchsize:", self.bs)
        print("Learning rate:", self.learning_rate)

        self.net = net
        print(net)

        self.criterion = criterion
        #  optimizer = optim.SGD(net.parameters(),
        #  lr=learning_rate,
        #  momentum=sgd_momentum)
        self.optimizer = optimizer(net.parameters(), lr=learning_rate)

    def train(self):
        trainloader = DataLoader(self.trainset,
                                 num_workers=self.num_w,
                                 batch_size=self.bs,
                                 shuffle=True)

        for epoch in tqdm(range(self.num_epochs)):
            self.net.train()

            running_loss = 0.0
            for _, (inputs, labels) in enumerate(trainloader, 0):

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                #  print(outputs.data)
                #  print(labels)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                #  print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

            if epoch % 10 == 9:
                #  print(outputs.data)
                #  print(labels)
                good_radio = self.test()
                #  if abs(good_radio - 1) < self.error_rate:
                    #  return epoch

    def test(self):
        """Test the DCNN."""

        print("Testing with :")
        print("Num workers:", self.num_w)
        print("Batchsize:", self.bs)

        # setting the classifier in test mode
        self.net.eval()
        #  W_1 = self.net.hid_layer.weight
        #  b_1 = self.net.hid_layer.bias
        #  W_2 = self.net.out_layer.weight
        #  b_2 = self.net.out_layer.bias
        #  print("W1 = {}\nb1 = {}\nW2 = {}\nb2={}".format(W_1, b_1, W_2, b_2))

        # loading data
        testloader = DataLoader(self.testset,
                                num_workers=self.num_w,
                                batch_size=self.bs,
                                shuffle=False)
        ngoodclassif = 0
        ntest = 0
        for _, (inputs, labels) in enumerate(testloader, 0):
            # forward in the net
            outputs = self.net(inputs)
            pred_correct = dict()
            # print(outputs)
            # print(labels)

            # compute the number of well
            # classified data and the total number of tests
            for inp, predicted, expected in zip(inputs, outputs, labels):
                # print(p, l, abs(p-l))
                #  predicted = torch.bernoulli(predicted)
                if abs(predicted - expected) < 0.5:
                    ngoodclassif += 1
                    pred_correct[inp] = True
                else:
                    pred_correct[inp] = False
                ntest += 1

        ratio = ngoodclassif / ntest
        print('Good classification ratio: %.2f %%' % (ratio * 100))
        print(pred_correct)

        return ratio
