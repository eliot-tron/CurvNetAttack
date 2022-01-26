import torch
import network as net
import foliation as foli
import adversarial_attack as advatt
import matplotlib.pyplot as plt


if __name__ == "__main__":
    print("hello, let's begin")
    objective = "xor"  # "xor"
    print("Objective: {}".format(objective))   
    # torch.manual_seed(0)
    if objective == "xor":
        testset = net.XorDataset(test=True)
        trainset = net.XorDataset(nsample=2000)
    elif objective == "or":
        testset = net.OrDataset(test=True)
        trainset = net.OrDataset(nsample=2000)
    # torch.manual_seed(0)
    our_net = net.XorNet(hid_size=150)
    trainer = net.Trainer(net=our_net, testset=testset, trainset=trainset, num_epochs=500)
    trainer.train()
    # TODO: Maybe do multiple network for more stable results <13-01-22> #
    # <24-01-22> Indeed: the fooling rates changes quite a bit #
    with torch.no_grad():
        foliation = foli.Foliation(our_net, objective)
        foliation.plot()
        size=1
        # torch.manual_seed(0)
        # OSSA = advatt.OneStepSpectralAttack(our_net, objective)
        # OSSA.plot_fooling_rates(step=0.05, size=size)
        # OSSA.plot_attacks(budget=0.2)
        # torch.manual_seed(0)
        # STSSA = advatt.StandardTwoStepSpectralAttack(our_net, objective)
        # STSSA.plot_attacks(budget=0.1)
        # torch.manual_seed(0)
        # TSSA = advatt.TwoStepSpectralAttack(our_net, objective)
        # TSSA.plot_fooling_rates(step=0.05, size=size)
        # TSSA.plot_attacks(budget=0.1)
        # savepath = "./plots/fooling_rates_compared_{}_size={}".format(objective, size)
        # plt.legend()
        # plt.savefig(savepath + '.pdf', format='pdf')
        # plt.show()
