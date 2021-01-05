import os
import json
import numpy as np


class Reporter(object):

    """Bridging between in-model evaluate and in-trainer logger.

    How to use:
        1) Initialize Reporter in model.__init__()
        2) Call Reporter.report() in model's loss calculation.
    """

    def __init__(self, print_keys=None):
        self.observation = {
            'epoch': [0],
            'iteration': [0],
        }
        self.epoch = 0
        self.iteration = 0
        self.triggers = None
        self.phase = 'main'
        self.print_keys = print_keys

    def set_phase(self, phase_name):
        self.phase = phase_name

    def set_intervals(self, triggers_dict):
        self.triggers = triggers_dict

    def report(self, report_dict):
        for k, v in report_dict.items():
            key_name = self.phase + '/' + k

            if key_name in self.observation:
                self.observation[key_name].append(v)
            else:
                self.observation[key_name] = [v]

    def print_report(self, out_dir):
        if self.phase != 'main':
            return

        trigger = self.triggers['report_trigger']

        if (self.observation[trigger.get_unit()][-1]
            %trigger.get_number()==0):

            print_keys = self.print_keys
            if not print_keys:
                print("\t".join([
                    k+": "+str(self.observation[k][-1])
                    for k
                    in ['epoch', 'iteration']
                ]))
            else:
                ei = [
                    k+": "+str(self.observation[k][-1])
                    for k
                    in ['epoch', 'iteration']
                ]
                normalize_standard = trigger.get_unit()
                if normalize_standard == 'epoch':
                    norm_range = \
                        np.where(
                            np.array(self.observation['epoch'])==self.observation['epoch'][-1]
                        )[0]
                    range_start = norm_range[0]
                elif normalize_standard == 'iteration':
                    range_start = - trigger.get_number()
                kv = [
                    k+": {:.5f}".format(np.mean(self.observation[k][range_start:]))
                    for k
                    in print_keys
                ]
                print("\t".join(ei + kv))

    def log_report(self, out_dir):
        with open(os.path.join(out_dir, 'log.json'), 'w') as fw:
            json.dump(self.observation, fw, indent=4)

    def check_save_trigger(self):
        trigger = self.triggers['save_trigger']
        if (self.observation[trigger.get_unit()][-1]
            %trigger.get_number()==0):
            return True
        else:
            return False

    def check_log_trigger(self):
        trigger = self.triggers['log_trigger']
        if (self.observation[trigger.get_unit()][-1]
            %trigger.get_number()==0):
            return True
        else:
            return False

    def check_eval_trigger(self):
        trigger = self.triggers['eval_trigger']
        if (self.observation[trigger.get_unit()][-1]
            %trigger.get_number()==0):
            return True
        else:
            return False

    def check_stop_trigger(self):
        trigger = self.triggers['stop_trigger']
        if (self.observation[trigger.get_unit()][-1]==trigger.get_number()):
            return False
        else:
            return True

    def count_iter(self):
        self.iteration += 1
        self.observation['iteration'].append(self.iteration)
        self.observation['epoch'].append(self.epoch)

    def count_epoch(self):
        self.epoch += 1
