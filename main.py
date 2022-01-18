import torch
import network as net
import foliation as foli
import adversarial_attack as advatt


if __name__ == "__main__":
    print("hello, let's begin")
    objective = "xor"  # "xor"
    print("Objective: {}".format(objective))
    if objective == "xor":
        testset = net.XorDataset(test=True)
        trainset = net.XorDataset(nsample=2000)
    elif objective == "or":
        testset = net.OrDataset(test=True)
        trainset = net.OrDataset(nsample=2000)
    our_net = net.XorNet(hid_size=150)
    trainer = net.Trainer(net=our_net, testset=testset, trainset=trainset, num_epochs=50)
    trainer.train()
    # TODO: Maybe do multiple network for more stable results <13-01-22> #
    with torch.no_grad():
        #  foliation = foli.Foliation(our_net, objective)
        #  foliation.plot()
        OSSA = advatt.OneStepSpectralAttack(our_net, objective)
        OSSA.plot_fooling_rates(step=0.05)
        TSSA = advatt.TwoStepSpectralAttack(our_net, objective)
        TSSA.plot_fooling_rates(step=0.05)
