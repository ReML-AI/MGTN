import os
import shutil
import time
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchnet as tnt
import torchvision.transforms as transforms
import torch.nn as nn
from util import *
from logger import *

tqdm.monitor_interval = 0
class MultiLabelEngine(object):
    def __init__(self, state={}):
        self.state = {
            'use_gpu': torch.cuda.is_available(),
            'image_size': 224,
            'batch_size': 64,
            'batch_size_test': 64,
            'workers': 25,
            'device_ids': None,
            'evaluate': False,
            'lr_decay': 0.1,
            'start_epoch': 0,
            'max_epochs': 300,
            'epoch_step': [],
            'difficult_examples': False,
            'mlt': 0.999,
            'use_pb': True,
            'print_freq': 0,
            'arch': '',
            'resume': None,
            'save_model_path': None,
            'filename_previous_best': None,
            'meter_loss': tnt.meter.AverageValueMeter(),
            'batch_time': tnt.meter.AverageValueMeter(),
            'data_time': tnt.meter.AverageValueMeter(),
            'best_score': 0
        }
        self.state.update(state)
        self.state.setdefault('train_transform', transforms.Compose([
                transforms.Resize((512, 512)),
                MultiScaleCrop(self.state['image_size'], scales=(1.0, 0.875, 0.75, 0.66, 0.5), max_distort=2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])]))
        self.state.setdefault('train_target_transform', None)
        self.state.setdefault('val_transform', transforms.Compose([
                Warp(self.state['image_size']),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])]))
        self.state.setdefault('val_target_transform', None)
        self.state['ap_meter'] = AveragePrecisionMeter(self.state['difficult_examples'])

    def on_start_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        self.state['meter_loss'].reset()
        self.state['batch_time'].reset()
        self.state['data_time'].reset()
        self.state['ap_meter'].reset()

    def on_end_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        map = 100 * self.state['ap_meter'].value().mean()
        loss = self.state['meter_loss'].value()[0]
        OP, OR, OF1, CP, CR, CF1 = self.state['ap_meter'].overall()
        OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = self.state['ap_meter'].overall_topk(3)
        if display:
            if training:
                print('Epoch: [{0}]\t Loss {loss:.4f}\t mAP {map:.3f}'.format(self.state['epoch'], loss=loss, map=map))
                print('OP: {OP:.4f}\t OR: {OR:.4f}\t OF1: {OF1:.4f}\t CP: {CP:.4f}\t CR: {CR:.4f}\t CF1: {CF1:.4f}'
                      .format(OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1))
                send_log('epoch', self.state['epoch'])
                send_log('train_loss', loss)
                send_log('train_map', map)
            else:
                print('Test: \t Loss {loss:.4f}\t mAP {map:.3f}'.format(loss=loss, map=map))
                print('OP: {OP:.4f}\t OR: {OR:.4f}\t OF1: {OF1:.4f}\t CP: {CP:.4f}\t CR: {CR:.4f}\t CF1: {CF1:.4f}'
                      .format(OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1))
                print('OP_3: {OP:.4f}\t OR_3: {OR:.4f}\t OF1_3: {OF1:.4f}\t CP_3: {CP:.4f}\t CR_3: {CR:.4f}\t CF1_3: {CF1:.4f}'
                      .format(OP=OP_k, OR=OR_k, OF1=OF1_k, CP=CP_k, CR=CR_k, CF1=CF1_k))
                send_log('test_loss', loss)
                send_log('test_map', map)
        return map

    def on_start_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        self.state['target_gt'] = self.state['target'].clone()
        self.state['target'][self.state['target'] == 0] = 1
        self.state['target'][self.state['target'] == -1] = 0

        input = self.state['input']
        self.state['img'] = input[0]
        self.state['name'] = input[1]


    def on_end_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        # record loss
        self.state['loss_batch'] = self.state['loss'].item()
        self.state['meter_loss'].add(self.state['loss_batch'])

        # measure mAP
        self.state['ap_meter'].add(self.state['output'].data, self.state['target_gt'])
        
        if display and self.state['print_freq'] != 0 and self.state['iteration'] % self.state['print_freq'] == 0:
            loss = self.state['meter_loss'].value()[0]
            batch_time = self.state['batch_time'].value()[0]
            data_time = self.state['data_time'].value()[0]
            if training:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'Data {data_time_current:.3f} ({data_time:.3f})\t'
                      'Loss {loss_current:.4f} ({loss:.4f})'.format(
                    self.state['epoch'], self.state['iteration'], len(data_loader),
                    batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    data_time=data_time, loss_current=self.state['loss_batch'], loss=loss))
            else:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'Data {data_time_current:.3f} ({data_time:.3f})\t'
                      'Loss {loss_current:.4f} ({loss:.4f})'.format(
                    self.state['iteration'], len(data_loader), batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    data_time=data_time, loss_current=self.state['loss_batch'], loss=loss))

    def on_forward(self, training, model, criterion, data_loader, optimizer=None, display=True):
        with torch.set_grad_enabled(training):
            img_var = torch.autograd.Variable(self.state['img']).float()
            target_var = torch.autograd.Variable(self.state['target']).float()
            # compute output
            self.state['output'] = model(img_var)
            self.state['loss'] = criterion(self.state['output'], target_var)
            if training:
                optimizer.zero_grad()
                self.state['loss'].backward()
                optimizer.step()
            else:
                torch.cuda.empty_cache()


    def learning(self, model, criterion, train_dataset, val_dataset, optimizer=None):

        # define train and val transform
        train_dataset.transform = self.state['train_transform']
        train_dataset.target_transform = self.state['train_target_transform']
        val_dataset.transform = self.state['val_transform']
        val_dataset.target_transform = self.state['val_target_transform']

        # data loading code
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=self.state['batch_size'], shuffle=True,
                                                   num_workers=self.state['workers'])

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=self.state['batch_size_test'], shuffle=False,
                                                 num_workers=self.state['workers'])

        # optionally resume from a checkpoint
        if self.state['resume'] is not None:
            if os.path.isfile(self.state['resume']):
                print("=> loading checkpoint '{}'".format(self.state['resume']))
                checkpoint = torch.load(self.state['resume'])
                self.state['start_epoch'] = checkpoint['epoch']
                self.state['best_score'] = checkpoint['best_score']
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(self.state['evaluate'], checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(self.state['resume']))


        if self.state['use_gpu']:
            train_loader.pin_memory = True
            val_loader.pin_memory = True
            cudnn.benchmark = False

            model = torch.nn.DataParallel(model, device_ids=self.state['device_ids']).cuda()

            criterion = criterion.cuda()

        if self.state['evaluate']:
            self.validate(val_loader, model, criterion)
            return

        for epoch in range(self.state['start_epoch'], self.state['max_epochs']):
            self.state['epoch'] = epoch
            lr = self.adjust_learning_rate(optimizer)
            print('lr:',lr, '|', 'step:' ,self.state['epoch_step'],'|', 'decay: ', self.state['lr_decay'])

            # train for one epoch
            self.train(train_loader, model, criterion, optimizer, epoch)
            # evaluate on validation set
            prec1 = self.validate(val_loader, model, criterion)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > self.state['best_score']
            self.state['best_score'] = max(prec1, self.state['best_score'])
            self.save_checkpoint({
                'epoch': epoch + 1,
                'arch': self.state['arch'],
                'state_dict': model.module.state_dict() if self.state['use_gpu'] else model.state_dict(),
                'best_score': self.state['best_score'],
            }, is_best)

            print(' *** best={best:.3f}'.format(best=self.state['best_score']))
            set_log_property('top', float(self.state['best_score']))

        return self.state['best_score']

    def run(self, training, data_loader, model, criterion, optimizer=None, epoch=None):
        if training:
            # switch to train mode
            model.train()
        else:
            # switch to evaluate mode
            model.eval()

        self.on_start_epoch(training, model, criterion, data_loader, optimizer)

        if self.state['use_pb']:
            data_loader = tqdm(data_loader, desc='Training' if training else 'Test')

        end = time.time()
        for i, (input, target) in enumerate(data_loader):
            # measure data loading time
            self.state['iteration'] = i
            self.state['data_time_batch'] = time.time() - end
            self.state['data_time'].add(self.state['data_time_batch'])

            self.state['input'] = input
            self.state['target'] = target

            self.on_start_batch(training, model, criterion, data_loader, optimizer)

            if self.state['use_gpu']:
                self.state['target'] = self.state['target'].cuda()

            self.on_forward(training, model, criterion, data_loader, optimizer)

            # measure elapsed time
            self.state['batch_time_current'] = time.time() - end
            self.state['batch_time'].add(self.state['batch_time_current'])
            end = time.time()
            # measure accuracy
            self.on_end_batch(training, model, criterion, data_loader, optimizer)

        return self.on_end_epoch(training, model, criterion, data_loader, optimizer)

    def train(self, data_loader, model, criterion, optimizer, epoch):
        return self.run(True, data_loader, model, criterion, optimizer, epoch)

    def validate(self, data_loader, model, criterion):
        return self.run(False, data_loader, model, criterion)

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        if self.state['save_model_path'] is not None:
            filename_ = filename
            filename = os.path.join(self.state['save_model_path'], filename_)
            if not os.path.exists(self.state['save_model_path']):
                os.makedirs(self.state['save_model_path'])
        print('save model {filename}'.format(filename=filename))
        torch.save(state, filename)
        if is_best:
            filename_best = 'model_best.pth.tar'
            if self.state['save_model_path'] is not None:
                filename_best = os.path.join(self.state['save_model_path'], filename_best)
            shutil.copyfile(filename, filename_best)
            if self.state['save_model_path'] is not None:
                if self.state['filename_previous_best'] is not None:
                    os.remove(self.state['filename_previous_best'])
                filename_best = os.path.join(self.state['save_model_path'], 'model_best_{score:.4f}.pth.tar'.format(score=state['best_score']))
                shutil.copyfile(filename, filename_best)
                self.state['filename_previous_best'] = filename_best

    def adjust_learning_rate(self, optimizer):
        """Sets the learning rate to the initial LR decayed by a fraction every epoch steps"""
        lr_list = []
        decay = self.state['lr_decay'] if sum(self.state['epoch'] == np.array(self.state['epoch_step'])) > 0 else 1.0
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay
            lr_list.append(param_group['lr'])
        return np.unique(lr_list)

class GraphMultiLabelEngine(MultiLabelEngine):
    def on_forward(self, training, model, criterion, data_loader, optimizer=None, display=True):
        with torch.set_grad_enabled(training):
            img_var = torch.autograd.Variable(self.state['img']).float()
            target_var = torch.autograd.Variable(self.state['target']).float()
            emb_var = torch.autograd.Variable(self.state['emb']).float().detach()  # one hot
            # compute output
            self.state['output'] = model(img_var, emb_var)
            self.state['loss'] = criterion(self.state['output'], target_var)
            if training:
                optimizer.zero_grad()
                self.state['loss'].backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
            else:
                torch.cuda.empty_cache()

    def on_start_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        MultiLabelEngine.on_start_batch(self, training, model, criterion, data_loader, optimizer, display)
        self.state['emb'] = self.state['input'][2]

