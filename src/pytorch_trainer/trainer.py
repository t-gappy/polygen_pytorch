import os
import time
import torch
import numpy as np
from .reporter import Reporter


class Trigger(object):

    """Trigger class to interpret epoch/iteration of user-defined event.

    args:
        trigger_tuple [tuple(int, str)]: trigger to user-defined event.
                (1, 'epoch') means 1epoch for event.
    """

    def __init__(self, trigger_tuple):
        self.number, self.unit = trigger_tuple

        if self.unit not in ['epoch', 'iteration']:
            raise ValueError('trigger must be (int, `epoch`/`iteration`)')

    def get_number(self):
        return self.number

    def get_unit(self):
        return self.unit


class Trainer(object):

    """chainer-like(only mimic) trainer class for pytorch.

    args:
        model [nn.Module]: model class to train.
        optimizer [torch.optimizer]: optimizer class to train the model.
        loaders [list(DataLoader)]: DataLoader used in train/validation.
                This list takes 1 or 2 DataLoader object.
                If 1 element exists in list, no validation was carried out.
                If 2 element exist, first one for train, second one for validation.
        reporter [Reporter]: Reporter class to bridging model and trainer.
                When this arg takes `None`, reporter was initialized in trainer.
                But, no `print_keys` arg in Reporter will be specified.
                (So only `epoch` and `iteration` were reported.)
        gpu [bool]: whether or not to use gpu in training.
        device_id [int]: specified gpu id to use.
        stop_trigger [Trigger]: when to training end.
        save_trigger [Trigger]: intervals to save checkpoints.
        report_trigger [Trigger]: intervals to report Reporter's observation.
        out_dir [str]: directory path for output.
    """

    def __init__(self, model, optimizer, loaders, ckpt_path=None,
                 reporter=None, gpu=None, device_id=None,
                 stop_trigger=(1, 'epoch'), save_trigger=(1, 'epoch'),
                 log_trigger=(1, 'epoch'), eval_trigger=(1, 'epoch'),
                 report_trigger=(10, 'iteration'), out_dir='./'):

        if len(loaders) == 2:
            self.eval_in_train = True
        else:
            self.eval_in_train = False

        if gpu == "gpu" and torch.cuda.is_available():
            if device_id is None:
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cuda:{}'.format(device_id))
            model = model.cuda(self.device)
        else:
            self.device = None

        if reporter is None:
            reporter = Reporter()

        trigger_dict = {'stop_trigger': Trigger(stop_trigger),
                        'save_trigger': Trigger(save_trigger),
                        'report_trigger': Trigger(report_trigger),
                        'log_trigger': Trigger(log_trigger),
                        'eval_trigger': Trigger(eval_trigger)}
        reporter.set_intervals(trigger_dict)
        model.reporter = reporter

        self.model = model
        self.optimizer = optimizer
        self.loaders = loaders
        self.out_dir = out_dir
        
        if ckpt_path:
            self._load_checkpoint(ckpt_path)

    def run(self):
        """Training loops for epoch.
        """
        model = self.model
        optimizer = self.optimizer
        loaders = self.loaders
        eval_in_train = self.eval_in_train
        device = self.device

        while model.reporter.check_stop_trigger():
            try:

                model.reporter.set_phase('main')
                model.train()
                for i, batch in enumerate(loaders[0]):
                    isnan, error_batch = self._update(model, optimizer, batch, device)
                    if isnan:
                        with open(self.out_dir+"error_log.txt", "a") as fa:
                            print("batch number: ", i, file=fa)
                            print(batch, file=fa)

                    model.reporter.print_report(self.out_dir)
                    model.reporter.count_iter()

                if eval_in_train and model.reporter.check_eval_trigger():
                    model.reporter.set_phase('validation')
                    model.eval()
                    with torch.no_grad():
                        for batch in loaders[1]:
                            self._evaluate(model, batch, device)

                model.reporter.count_epoch()
                if model.reporter.check_log_trigger():
                    model.reporter.log_report(self.out_dir)
                if model.reporter.check_save_trigger():
                    self._save_checkpoint(model)

            except KeyboardInterrupt:
                model.reporter.log_report(self.out_dir)
                raise KeyboardInterrupt

        model.reporter.log_report(self.out_dir)


    def _update(self, model, optimizer, batch, device):
        optimizer.zero_grad()
        loss = model(*batch, device=device)
        if np.isnan(loss.item()):
            return True, batch
        loss.backward()
        optimizer.step()
        return False, None


    def evaluate(self):
        """Function for evaluation after training.
        """
        return


    def _evaluate(self, model, batch, device):
        loss = model(*batch, device=device)


    def _save_checkpoint(self, model):
        epoch_num = model.reporter.observation['epoch'][-1]
        file_name = os.path.join(self.out_dir, 'model_epoch_{}'.format(epoch_num))
        state = {
            'epoch': epoch_num+1,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, file_name)
        
    def _load_checkpoint(self, file_name):
        ckpt = torch.load(file_name)
        self.model.load_state_dict(ckpt['state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        print("restart from", ckpt['epoch'], 'epoch.')
