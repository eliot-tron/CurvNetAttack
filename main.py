import torch
import network as net
import foliation as foli
import adversarial_attack as advatt
import matplotlib.pyplot as plt
from tqdm import tqdm


if __name__ == "__main__":
    print("hello, let's begin")
    objective = "xor"  # "or"
    print("Objective: {}".format(objective))   
    # torch.use_deterministic_algorithms(True)
    # torch.manual_seed(0)
    if objective == "xor":
        testset = net.XorDataset(test=True)
        trainset = net.XorDataset(nsample=5000)
    elif objective == "or":
        testset = net.OrDataset(test=True)
        trainset = net.OrDataset(nsample=5000)
    # torch.manual_seed(0)
    our_net = net.XorNet(hid_size=50)
    trainer = net.Trainer(net=our_net, testset=testset, trainset=trainset, num_epochs=100)
    trainer.train()
    # TODO: Maybe do multiple network for more stable results <13-01-22> #
    # <24-01-22> Indeed: the fooling rates changes quite a bit #
    with torch.no_grad():
        foliation = foli.Foliation(our_net, objective)
        foliation.plot(eigenvectors=False, transverse=True)
        # OSSA = advatt.OneStepSpectralAttack(our_net, objective)
        # TSSA = advatt.TwoStepSpectralAttack(our_net, objective)
        # STSSA = advatt.StandardTwoStepSpectralAttack(our_net, objective)
        # for size in tqdm(range(1, 11)):
        #     size /= 10
        #     torch.manual_seed(0)
        #     OSSA.plot_fooling_rates(step=0.01, size=size, end=0.5)
        #     torch.manual_seed(0)
        #     STSSA.plot_fooling_rates(step=0.01, size=size, end=0.5)
        #     torch.manual_seed(0)
        #     TSSA.plot_fooling_rates(step=0.01, size=size, end=0.5)
        #     savepath = "./plots/fooling_rates_compared_{}_size={}".format(objective, size)
        #     plt.legend()
        #     plt.savefig(savepath + '.pdf', format='pdf')
        #     plt.clf()
        # torch.manual_seed(0)
        # OSSA.plot_attacks(budget=0.2)
        # torch.manual_seed(0)
        # STSSA.plot_attacks(budget=0.1)
        # torch.manual_seed(0)
        # TSSA.plot_attacks(budget=0.1)

