import numpy as np
from simulator import SimulationDataset
from torch.utils.data import DataLoader
from utils import EarlyStopping, cal_NLL, cal_AUSE, cal_AUCE, inf_plot_1D, inf_plot_ND, cal_95CR
import torch
import copy
import os
import matplotlib.pyplot as plt
import seaborn as sns


class Base_Trainer(object):
    def __init__(self, config, base_net, ensembles, criterion, evaluation,
                 TrainX, TrainY, TestX, TestY, EvalX, EvalY, writer, logger, device):
        self.config = config
        self.base_net = base_net
        self.base_net_ckpts = None
        self.ensembles = ensembles
        self.ensembles_ckpts = []
        self.criterion = criterion
        self.evaluation = evaluation
        self.TrainX = TrainX
        self.TrainY = TrainY
        self.TestX = TestX
        self.TestY = TestY
        self.EvalX = EvalX
        self.EvalY = EvalY
        self.writer = writer
        self.logger = logger
        self.device = device
        self.train_loaders = []

        self.single_train_loader = None
        self.test_loader = []
        train_set = SimulationDataset(self.TrainX, self.TrainY)
        test_set = SimulationDataset(self.TestX, self.TestY)
        self.num_evalX = config.num_xeval
        eval_set = SimulationDataset(self.EvalX, self.EvalY)

        self.single_train_loader = DataLoader(train_set, batch_size=self.config.batch_size, shuffle=True, num_workers=0,
                                              drop_last=True,
                                              pin_memory=True)
        self.test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False, num_workers=0,
                                      drop_last=False,
                                      pin_memory=True)
        self.eval_loader = DataLoader(eval_set, batch_size=self.num_evalX, shuffle=False, num_workers=0,
                                      drop_last=False,
                                      pin_memory=True)

        f = os.path.join(self.config.root, self.config.dataset, 'EvalY_gt.npz')
        self.EvalY_mu_gt = np.load(f)['Y_mu']
        self.EvalY_var_gt = np.load(f)['Y_var']
        self.EvalY_ent_gt = np.load(f)['Y_ent']

        self.m_var_path = os.path.join(self.config.root, self.config.dataset, self.config.cls_model, 'EvalM_gt.npz')
        if os.path.isfile(self.m_var_path):
            self.EvalM_var_gt = np.load(self.m_var_path)['M_var']
        elif self.config.method == 'Truth':
            self.EvalM_var_gt = np.nan
        else:
            self.EvalM_var_gt = np.zeros_like(self.EvalY_var_gt)

        subset_idx = np.arange(0, len(EvalY), step=10)
        eval_subset = SimulationDataset(self.EvalX[subset_idx], self.EvalY[subset_idx])
        self.Eval_subY_mu_gt = np.load(f)['Y_mu'][subset_idx]
        self.Eval_subY_var_gt = np.load(f)['Y_var'][subset_idx]
        self.eval_subset_loader = DataLoader(eval_subset, batch_size=self.num_evalX, shuffle=False, num_workers=0, drop_last=False,pin_memory=True)

        self.n = len(self.TrainY)
        self.n1, self.n2 = round(0.4 * self.n), round(0.6 * self.n)
        self.train_loaders1, self.train_loaders2 = [], []
        self.ensembles1, self.ensembles2 = copy.deepcopy(self.ensembles), copy.deepcopy(self.ensembles)

        # # adv
        # self.use_adv = False
        # self.config.eps = 1
        # self.config.alpha = 0.1
        # self.attack = FastGradientSignUntargeted(self.config.eps, self.config.alpha, min_val=0, max_val=10, max_iters=1)
        # self.adv_w = 0.2

    def __setup(self):
        raise NotImplementedError

    def single_train(self):
        early_stop = EarlyStopping()
        best_rmse = np.inf
        mean_mae_arr, mean_absrel_arr, var_mae_arr, var_absrel_arr, conv_epoch = [], [], [], [], []
        for epoch in range(self.config.epoch):
            # Train
            self.base_net.train()
            train_loss = []
            for i, (x, y_target) in enumerate(self.single_train_loader):
                self.base_net.zero_grad()
                self.base_net.optim.zero_grad()
                x, y_target = x.to(self.device), y_target.to(self.device)
                output = self.base_net(x)
                loss = self.criterion(output, y_target)
                loss.backward()
                self.base_net.optim.step()
                train_loss.append(loss.item())

            # Test
            self.base_net.eval()
            self.base_net.zero_grad()
            self.base_net.optim.zero_grad()
            test_loss, test_rmse, test_mu, test_var, test_ent = \
                self.single_test(self.base_net, self.eval_subset_loader)
        #
        #     mean_absrel = np.mean(np.abs(test_mu-self.Eval_subY_mu_gt)/self.Eval_subY_mu_gt)
        #     var_absrel = np.mean(np.abs(test_var - self.Eval_subY_var_gt) / self.Eval_subY_var_gt)
        #     var_mae = np.mean(np.abs(test_mu-self.Eval_subY_mu_gt))
        #     mean_mae = np.mean(np.abs(test_var - self.Eval_subY_var_gt))
        #
        #     conv_epoch.append(epoch)
        #     mean_mae_arr.append(mean_mae)
        #     mean_absrel_arr.append(mean_absrel)
        #     var_mae_arr.append(var_mae)
        #     var_absrel_arr.append(var_absrel)
        #     # Output
        #     print('Epoch {:04d} | Mean AbsRel {:.4f} | Var AbsRel {:.4f}'.format(epoch, mean_mae, var_mae))
        #
        #     # if epoch > 100:
        # #     #     early_stop(var_absrel, self.base_net, epoch)
        #     if epoch%100 == 0:
        #         self.single_inference(epoch=epoch)
        #     # if early_stop.early_stop:
        #     #     print("Early stopping, best epoch: {:04d}, rmse: {:.4f}".format(early_stop.best_epoch, early_stop.best_rmse))
        #     #     break
        #
        # f = os.path.join(self.config.savingroot, 'Converge.npz')
        # np.savez(f, **{'Mean_mae': np.stack(mean_mae_arr), 'Mean_absrel': np.stack(mean_absrel_arr),
        #                'Var_mae': np.stack(var_mae_arr), 'Var_absrel': np.stack(var_absrel_arr), 'Epoch': np.array(conv_epoch)})
        # # reload checkpoints
        print("Training stopping, best epoch: {:04d}, rmse: {:.4f}".format(early_stop.best_epoch, early_stop.best_rmse))

        # self.base_net_ckpts = copy.deepcopy(early_stop.model_ckpts)
        # self.base_net.load_state_dict(copy.deepcopy(early_stop.model_ckpts))
        # torch.save(self.base_net.state_dict(), os.path.join(self.config.savingroot, 'ckpts/m0.pth'))

    def single_test(self, net, test_loader, var_method='definition', smooth=False, mc=False):
        # (y_hat, residual) = self.cal_residuals(net) if var_method == 'residual' else (None, None)
        net.eval()
        if mc:
            for m in net.modules():
                if m.__class__.__name__.startswith('Dropout'):
                    m.train()

        test_loss, test_rmse = [], []
        test_mu, test_var, test_ent = np.array([]), np.array([]), np.array([])
        with torch.no_grad():
            for i, (x, y_target) in enumerate(test_loader):
                x, y_target = x.to(self.device), y_target.to(self.device)
                output = net(x)
                loss = self.criterion(output, y_target)
                metrics = self.evaluation(
                    output=output,
                    gt=y_target,
                    var_method=var_method,
                    smooth=smooth,
                    TrainX=self.TrainX,
                    TestX=x,
                    # residual=residual
                )
                test_loss.append(loss.item())
                test_rmse.append(metrics['rmse'])
                test_mu = np.concatenate((test_mu, metrics['mu_array']), axis=0)
                test_var = np.concatenate((test_var, metrics['var_array']), axis=0)
                test_ent = np.concatenate((test_ent, metrics['ent_array']), axis=0)
        return np.mean(test_loss), np.mean(test_rmse), test_mu, test_var, test_ent

    def single_inference(self, epoch, smooth=False):
        eval_loss, eval_rmse, eval_mu, eval_var, eval_ent = \
            self.single_test(self.base_net, self.eval_loader, var_method='definition', smooth=smooth)
        self.inference(mu=eval_mu, var_y=eval_var, ent_y=eval_ent, var_m=np.zeros(eval_var.shape), epoch=epoch)

    def cal_residuals(self, net):
        # rebuild dataset
        train_set = SimulationDataset(self.TrainX, self.TrainY)
        train_loader = DataLoader(train_set, batch_size=self.config.batch_size, shuffle=False, num_workers=0,
                                  drop_last=True,
                                  pin_memory=True)
        # get yhat and residual
        y_hat = np.array([])
        net.eval()
        for i, (x, y_target) in enumerate(train_loader):
            net.zero_grad()
            net.optim.zero_grad()
            x, y_target = x.to(self.device), y_target.to(self.device)
            y_probs = net(x)
            net.optim.step()
            metrics = self.evaluation(y_probs, y_target)
            y_hat = np.concatenate((y_hat, metrics['mu_array']), axis=0)

        res = self.TrainY - y_hat
        return y_hat, res

    def train(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    def ensembles_step(self, net, x, y_target):
        net.zero_grad()
        net.optim.zero_grad()
        x, y_target = x.to(self.device), y_target.to(self.device)
        y_probs = net(x)
        loss = self.criterion(y_probs, y_target)
        loss.backward()
        net.optim.step()
        return loss


    def ensembles_train(self):
        for m in range(self.config.num_ensemble):
            print("Net #{} training:".format(m))
            net = self.ensembles[m]
            train_loader = self.train_loaders[m]
            early_stop = EarlyStopping()
            for epoch in range(self.config.epoch):
                # train
                net.train()
                train_loss = []
                for i, (x, y_target) in enumerate(train_loader):
                    loss = self.ensembles_step(net, x, y_target)
                    train_loss.append(loss.item())
            #     # test
            #     net.eval()
            #     net.zero_grad()
            #     net.optim.zero_grad()
            #     test_loss, test_rmse, test_mu, test_var, test_ent = \
            #         self.single_test(net, self.test_loader)
            #     # print('Net #{} Epoch {:04d} | Train_Loss {:.4f} | Test_Loss {:.4f}| Test_Rmse {:.4f} '.format(
            #     #     m, epoch, np.mean(train_loss), np.mean(test_loss), np.mean(test_rmse)))
            #     early_stop(test_rmse, net, epoch)
            #     if early_stop.early_stop:
            #         print("Net #{} Early stopping, best epoch: {:04d}, rmse: {:.4f}".format(m, early_stop.best_epoch, early_stop.best_rmse))
            #         break
            # net.load_state_dict(copy.deepcopy(early_stop.model_ckpts))
            # # torch.save(net.state_dict(), os.path.join(self.config.savingroot, 'ckpts/m{}.pth'.format(m+1)))
            # self.logger.log(m, early_stop.best_rmse, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
        print("Ensemble training finished!")

    def mc_inference(self):
        mc_eval_mu_arr = np.array([])
        mc_eval_var_arr = np.array([])
        mc_eval_ent_arr = np.array([])
        for _ in range(self.config.num_ensemble):
            mc_eval_loss, mmc_eval_rmse, mc_eval_mu, mc_eval_var, mc_eval_ent = \
                self.single_test(self.base_net, self.eval_loader, mc=True)
            mc_eval_mu = mc_eval_mu.reshape(-1, len(self.EvalY))
            mc_eval_var = mc_eval_var.reshape(-1, len(self.EvalY))
            mc_eval_ent = mc_eval_ent.reshape(-1, len(self.EvalY))
            mc_eval_mu_arr = np.vstack((mc_eval_mu_arr, mc_eval_mu)) if mc_eval_mu_arr.size else mc_eval_mu
            mc_eval_var_arr = np.vstack((mc_eval_var_arr, mc_eval_var)) if mc_eval_var_arr.size else mc_eval_var
            mc_eval_ent_arr = np.vstack((mc_eval_ent_arr, mc_eval_ent)) if mc_eval_ent_arr.size else mc_eval_ent

        eval_mu_bar = np.mean(mc_eval_mu_arr, axis=0)
        eval_vary_bar = np.mean(mc_eval_var_arr, axis=0)
        eval_enty_bar = np.mean(mc_eval_ent_arr, axis=0)
        eval_model_var = np.var(mc_eval_mu_arr, axis=0)

        self.inference(eval_mu_bar, eval_vary_bar, eval_enty_bar, eval_model_var, self.config.epoch)
        print("MC inference finished!")

    def ensembles_inference(self):
        ensembles_eval_mu = np.array([])
        ensembles_eval_var = np.array([])
        ensembles_eval_ent = np.array([])
        for m in range(self.config.num_ensemble):
            net = self.ensembles[m]
            model_eval_loss, model_eval_rmse, model_eval_mu, model_eval_var, model_eval_ent = \
                self.single_test(net, self.eval_loader)
            model_eval_mu = model_eval_mu.reshape(-1, len(self.EvalY))
            model_eval_var = model_eval_var.reshape(-1, len(self.EvalY))
            model_eval_ent = model_eval_ent.reshape(-1, len(self.EvalY))
            ensembles_eval_mu = np.vstack(
                (ensembles_eval_mu, model_eval_mu)) if ensembles_eval_mu.size else model_eval_mu
            ensembles_eval_var = np.vstack(
                (ensembles_eval_var, model_eval_var)) if ensembles_eval_var.size else model_eval_var
            ensembles_eval_ent = np.vstack(
                (ensembles_eval_ent, model_eval_ent)) if ensembles_eval_ent.size else model_eval_ent

        eval_mu_bar = np.mean(ensembles_eval_mu, axis=0)
        eval_vary_bar = np.mean(ensembles_eval_var, axis=0)
        eval_enty_bar = np.mean(ensembles_eval_ent, axis=0)
        eval_model_var = np.var(ensembles_eval_mu, axis=0)

        if self.config.method == 'Truth':
            self.EvalM_var_gt = eval_model_var
            np.savez(self.m_var_path, **{'X': self.EvalX, 'M_var': self.EvalM_var_gt})
            print("Truth finished, model variance saved.")
        else:
            self.inference(eval_mu_bar, eval_vary_bar, eval_enty_bar, eval_model_var, self.config.epoch)
            print("Ensemble inference finished!")

    def bootstrap_inference(self):
        ensembles_eval_mu = np.array([])
        for m in range(self.config.num_ensemble):
            net = self.ensembles[m]
            model_eval_loss, model_eval_rmse, model_eval_mu, model_eval_var, model_eval_ent = self.single_test(net, self.eval_loader)
            model_eval_mu = model_eval_mu.reshape(-1, len(self.EvalY))
            ensembles_eval_mu = np.vstack(
                (ensembles_eval_mu, model_eval_mu)) if ensembles_eval_mu.size else model_eval_mu
        eval_model_var = np.var(ensembles_eval_mu, axis=0)

        eval_loss, eval_rmse, eval_mu_bar, eval_vary_bar, eval_enty_bar = \
            self.single_test(self.base_net, self.eval_loader, var_method='definition', smooth=False)

        self.inference(eval_mu_bar, eval_vary_bar, eval_enty_bar, eval_model_var, self.config.epoch)
        print("Bootstrap inference finished!")

    def inference(self, mu, var_y, ent_y, var_m, epoch):
        cover_rate_error = np.nan
        if self.config.X_C == 1:
            inf_plot_1D(
                y=self.EvalY,
                y_hat=mu,
                y_mu_gt=self.EvalY_mu_gt,
                y_var_hat=var_y,
                y_var_gt=self.EvalY_var_gt,
                y_ent_hat=ent_y,
                y_ent_gt=self.EvalY_ent_gt,
                m_var=var_m,
                x=self.EvalX,
                config=self.config,
                epoch=epoch,
                m_var_gt=self.EvalM_var_gt,
            )
            cover_rate_error = 0.95 - cal_95CR(x=self.EvalX, y_hat=mu, y_var=var_y, gt=self.EvalY, model_var=var_m)
        else:
            inf_plot_ND(y=self.EvalY, y_hat=mu, y_mu_gt=self.EvalY_mu_gt, y_var=var_y, y_var_gt=self.EvalY_var_gt, config=self.config, epoch=epoch)

        mu_e = self.EvalY_mu_gt[:self.num_evalX] - mu[:self.num_evalX]
        mu_re = mu_e / self.EvalY_mu_gt[:self.num_evalX]
        rmse = np.sqrt(np.mean(np.square(mu_e)))
        mae, absrel = np.mean(np.abs(mu_e)), np.mean(np.abs(mu_re))

        y_var_e = self.EvalY_var_gt[:self.num_evalX] - var_y[:self.num_evalX]
        y_var_re = y_var_e / self.EvalY_var_gt[:self.num_evalX]
        y_var_mae, y_var_absrel = np.mean(np.abs(y_var_e)), np.mean(np.abs(y_var_re))
        ause, ause_x, ause_y = cal_AUSE(x=self.EvalX, y_hat=mu, y_var=ent_y, gt=self.EvalY, config=self.config, epoch=self.config.epoch, model_var=var_m)
        nll = cal_NLL(x=self.EvalX[:self.num_evalX], y_hat=mu[:self.num_evalX], y_var=var_y[:self.num_evalX],
                      gt=self.EvalY[:self.num_evalX])

        y_ent_e = self.EvalY_ent_gt[:self.num_evalX] - ent_y[:self.num_evalX]
        y_ent_re = y_ent_e / self.EvalY_ent_gt[:self.num_evalX]
        y_ent_mae, y_ent_absrel = np.mean(np.abs(y_ent_e)), np.mean(np.abs(y_ent_re))

        m_var_e = self.EvalM_var_gt[:self.num_evalX] - var_m[:self.num_evalX]
        m_var_re = m_var_e / self.EvalM_var_gt[:self.num_evalX]
        m_var_mae, m_var_absrel = np.mean(np.abs(m_var_e)), np.mean(np.abs(m_var_re))


        f = os.path.join(self.config.savingroot, 'Eval_metrics.npz')
        np.savez(f, **{'X': self.EvalX[:self.num_evalX],
                       'Y_hat': mu[:self.num_evalX],
                       'Y_var_hat': var_y[:self.num_evalX],
                       'Y_ent_hat': ent_y[:self.num_evalX],
                       'M_var_hat': var_m[:self.num_evalX],
                       'NLL': nll,
                       'AUSE_x': ause_x,
                       'AUSE_y': ause_y,
                       'Y_e': mu_e,
                       'Y_re': mu_re,
                       '95CRE': cover_rate_error,
                       'Y_var_e': y_var_e,
                       'Y_var_re': y_var_re,
                       'Y_ent_e': y_ent_e,
                       'Y_ent_re': y_ent_re,
                       'M_var_e': m_var_e,
                       'M_var_re': m_var_re,
                       'Y_mu_gt': self.EvalY_mu_gt[:self.num_evalX],
                       'Y_var_gt': self.EvalY_var_gt[:self.num_evalX],
                       'Y_ent_gt': self.EvalY_ent_gt[:self.num_evalX],
                       'M_var_gt': self.EvalM_var_gt[:self.num_evalX],
                       'X_full': self.EvalX,
                       'Y_full': self.EvalY,
                       })

        self.logger.log(
            epoch=epoch,
            rmse=rmse,
            mae=mae,
            absrel=absrel,
            nll=np.mean(nll),
            # ause=ause,
            y_var_mae=y_var_mae,
            y_var_absrel=y_var_absrel,
            y_ent_mae=y_ent_mae,
            y_ent_absrel=y_ent_absrel,
            m_var_mae=m_var_mae,
            m_var_absrel=m_var_absrel,
            cre=np.mean(np.abs(cover_rate_error)),
        )

    def cbs_train(self):
        for net in self.ensembles1:
            net.load_state_dict(copy.deepcopy(self.base_net.state_dict()))
        for m in range(self.config.num_ensemble):
            net = self.ensembles1[m]
            train_loader = self.train_loaders1[m]
            for epoch in range(self.config.epoch):
                net.train()
                for i, (x, y_target) in enumerate(train_loader):
                    loss = self.ensembles_step(net, x, y_target)

        for net in self.ensembles2:
            net.load_state_dict(copy.deepcopy(self.base_net.state_dict()))
        for m in range(self.config.num_ensemble):
            net = self.ensembles2[m]
            train_loader = self.train_loaders2[m]
            for epoch in range(self.config.epoch):
                net.train()
                for i, (x, y_target) in enumerate(train_loader):
                    loss = self.ensembles_step(net, x, y_target)

    def cbs_test(self):
        eval_model_var = self.cbs_inference()
        eval_loss, eval_rmse, eval_mu_bar, eval_vary_bar, eval_enty_bar = \
            self.single_test(self.base_net, self.eval_loader, var_method='definition', smooth=False)

        self.inference(eval_mu_bar, eval_vary_bar, eval_enty_bar, eval_model_var, self.config.epoch)
        print("Bootstrap inference finished!")

    def cbs_inference(self):
        ensembles_eval_mu1 = np.array([])
        for m in range(self.config.num_ensemble):
            net = self.ensembles1[m]
            model_eval_loss, model_eval_rmse, model_eval_mu, model_eval_var, model_eval_ent = \
                self.single_test(net, self.eval_loader)
            model_eval_mu = model_eval_mu.reshape(-1, len(self.EvalY))
            ensembles_eval_mu1 = np.vstack(
                (ensembles_eval_mu1, model_eval_mu)) if ensembles_eval_mu1.size else model_eval_mu
        eval_model_var1 = np.var(ensembles_eval_mu1, axis=0)

        ensembles_eval_mu2 = np.array([])
        for m in range(self.config.num_ensemble):
            net = self.ensembles2[m]
            model_eval_loss, model_eval_rmse, model_eval_mu, model_eval_var, model_eval_ent = \
                self.single_test(net, self.eval_loader)
            model_eval_mu = model_eval_mu.reshape(-1, len(self.EvalY))
            ensembles_eval_mu2 = np.vstack(
                (ensembles_eval_mu2, model_eval_mu)) if ensembles_eval_mu2.size else model_eval_mu
        eval_model_var2 = np.var(ensembles_eval_mu2, axis=0)

        x_axis = self.EvalX[:, 0]
        fig, ax = plt.subplots()
        sns.scatterplot(x_axis, eval_model_var1, s=2, label='n1')
        sns.scatterplot(x_axis, eval_model_var2, s=2, label='n2')
        plt.xlabel('x')
        plt.ylabel('Var(m)')
        fig.savefig(self.config.savingroot + '/m_var_subset.png')
        plt.cla()
        plt.clf()
        plt.close()

        fig, ax = plt.subplots()
        sns.scatterplot(x_axis, eval_model_var1 / eval_model_var2, s=2, label='var1/var2')
        plt.xlabel('x')
        plt.ylabel('Var(m)')
        fig.savefig(self.config.savingroot + '/cbs_frac.png')
        plt.cla()
        plt.clf()
        plt.close()

        var_frac = eval_model_var1 / eval_model_var2
        n1_div_n2 = self.n1 / self.n2
        tao = np.log(var_frac) / np.log(n1_div_n2)
        print('Tao: ', tao)
        tao_bar = np.mean(tao)
        print('Tao_bar', tao_bar)
        n_div_n1, n_div_n2 = self.n / self.n1, self.n / self.n2
        m_var = eval_model_var2 * np.power(n_div_n2, tao)
        m_var = np.clip(m_var, a_min=0, a_max=np.ceil(np.max(self.EvalM_var_gt)))
        m_var_appr = (eval_model_var2 * np.power(n_div_n2, tao_bar) + eval_model_var1 * np.power(n_div_n1, tao_bar))/2

        fig, ax = plt.subplots()
        sns.scatterplot(x_axis, self.EvalM_var_gt, s=2, label='gt')
        sns.scatterplot(x_axis, m_var, s=2, label='est')
        plt.xlabel('x')
        plt.ylabel('Var(m)')
        fig.savefig(self.config.savingroot + '/m_var_cbs.png')
        plt.cla()
        plt.clf()
        plt.close()

        fig, ax = plt.subplots()
        sns.scatterplot(x_axis, self.EvalM_var_gt, s=2, label='gt')
        sns.scatterplot(x_axis,  m_var_appr, s=2, label='est')
        plt.xlabel('x')
        plt.ylabel('Var(m)')
        fig.savefig(self.config.savingroot + '/m_var_cbs_appr.png')
        plt.cla()
        plt.clf()
        plt.close()

        fig, ax = plt.subplots()
        sns.scatterplot(x_axis, tao, s=2, label='pw')
        sns.scatterplot(x_axis, np.repeat(tao_bar, len(x_axis)), s=1, label='bar')
        plt.xlabel('x')
        plt.ylabel('tao')
        fig.savefig(self.config.savingroot + '/tao.png')
        plt.cla()
        plt.clf()
        plt.close()

        return m_var
