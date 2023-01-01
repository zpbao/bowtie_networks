import torch
import os
import time
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from torch import optim
from torch.autograd import grad
from skimage.io import imsave
from torchvision.utils import save_image
from itertools import chain
from torch import nn

from torchvision import transforms
from torch.autograd import Variable

import torch.nn.functional as F

few_num = 240
few_k = 5

class GAN:
    """
    Base model for GAN models
    """
    def __init__(self,
                 gen_fn,
                 disc_fn,
                 classifier_fn,
                 feat_ext,
                 z_dim,
                 num_classes = 200,
                 lamb=0.,
                 opt_g=optim.Adam,
                 opt_d=optim.Adam,
                 opt_c=optim.Adam,
                 opt_d_args={'lr': 0.0002, 'betas': (0.5, 0.999)},
                 opt_g_args={'lr': 0.0002, 'betas': (0.5, 0.999)},
                 opt_c_args={'lr':0.00005, 'betas': (0.5, 0.999)},
                 update_g_every=5,
                 handlers=[],
                 scheduler_fn=None,
                 scheduler_args={},
                 use_cuda='detect'):
        assert use_cuda in [True, False, 'detect']
        if use_cuda == 'detect':
            use_cuda = True if torch.cuda.is_available() else False
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.update_g_every = update_g_every
        self.g = gen_fn
        self.d = disc_fn
        self.c = classifier_fn
        self.fe = feat_ext
        self.lamb = lamb
        self.beta = 0.0
        optim_g = opt_g(filter(lambda p: p.requires_grad,
                               self.g.parameters()), **opt_g_args)
        optim_d = opt_d(filter(lambda p: p.requires_grad,
                               self.d.parameters()), **opt_d_args)
        optim_c = opt_c(filter(lambda p: p.requires_grad,
                               self.c.parameters()), **opt_c_args)
        self.optim = {
            'g': optim_g,
            'd': optim_d,
            'c': optim_c,
        }
        # HACK: this is actually both the MI network
        # and G.

        self.scheduler = {}
        if scheduler_fn is not None:
            for key in self.optim:
                self.scheduler[key] = scheduler_fn(
                    self.optim[key], **scheduler_args)
        self.handlers = handlers
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.g.cuda()
            self.d.cuda()
            self.c.cuda()
            self.fe.cuda()
        self.last_epoch = 0

    def _get_stats(self, dict_, mode):
        stats = OrderedDict({})
        for key in dict_.keys():
            stats[key] = np.mean(dict_[key])
        return stats

    def sample_z(self, imgs, seed=None):
        """Return a sample z ~ p(z)"""
        self.c.eval()
        if self.use_cuda:
            imgs = imgs.cuda()
        self.fe.eval()
        res_f = self.fe.resnet(imgs)
        feat = self.c.embed(res_f)
        noise = torch.randn_like(feat).cuda()
        res = torch.clamp(feat + noise, -1, 1)
        return res

    def sample_z_from_f(self, f, seed=None):
        """Return a sample z ~ p(z)"""
        self.c.eval()
        if self.use_cuda:
            f = f.cuda()
        feat = self.c.embed(f)
        noise = torch.randn_like(feat).cuda()
        res = feat+noise
        return res

    def loss(self, prediction, target):
        if not hasattr(target, '__len__'):
            target = torch.ones_like(prediction)*target
            if prediction.is_cuda:
                target = target.cuda()
        loss = torch.nn.BCELoss()
        if prediction.is_cuda:
            loss = loss.cuda()
        return loss(prediction, target)
    
    def c_loss(self, score, target):
        loss = torch.nn.CrossEntropyLoss()
        if score.is_cuda:
            loss = loss.cuda()
        return loss(score, target)

    def _train(self):
        self.g.train()
        self.d.train()
        self.c.train()

    def _eval(self):
        self.g.eval()
        self.d.eval()
        self.c.eval()
        
    def test_network(self, whole_classes, num_classes, train_x, train_y, test_x, test_y):
        self.c.eval()
        batch_size = train_x.shape[0]

        train_feats = torch.Tensor(train_x)
        train_labels = torch.Tensor(train_y).view(-1,1)
        train_labels = train_labels.expand(train_labels.size(0), num_classes)
        train_Y = torch.arange(num_classes).view(1,-1)
        train_Y = train_Y.expand_as(train_labels)
        train_Y = (train_Y == train_labels).to(torch.float)
        test_feats = torch.Tensor(test_x)
        test_labels = torch.Tensor(test_y)
        count = 0

        train_feats = Variable(train_feats.cuda())
        train_Y = Variable(train_Y.cuda())
        test_feats = Variable(test_feats.cuda())
        test_labels = Variable(test_labels.long().cuda())

        logprob = self.c(train_feats, test_feats, train_Y)
        junk, amax = logprob.max(1)
        acc = np.mean((amax.data.cpu().numpy().reshape(-1) == test_labels.data.cpu().numpy().reshape(-1)))

        _,top5 = logprob.topk(5,1)
        acc5 = 0.0
        for tp in range(5):
            acc5 += np.mean(top5[:,tp].data.cpu().numpy().reshape(-1) == test_labels.data.cpu().numpy().reshape(-1))

        return acc, acc5 

    def few_test(self, train_x, train_y, test_x, test_y):
        self.c.eval()
        train_f = torch.zeros((self.num_classes - few_num)*few_k, 1000)
        train_Y = torch.zeros((self.num_classes - few_num)*few_k, (self.num_classes - few_num))
        count = 0
        for i in range(few_num,self.num_classes):
            idx = np.where(train_y==i)[0][:few_k]
            train_sample = torch.Tensor(train_x[list(idx)])
            train_f[count*few_k:count*few_k+few_k] = train_sample
            train_Y[count*few_k:count*few_k+few_k,count] = 1
            count+=1
        idx = np.where(test_y >= few_num)[0]
        test_f = torch.Tensor(test_x[list(idx)])
        test_yy = torch.Tensor(test_y[list(idx)]) - few_num
        
        train_feats = Variable(train_f.cuda())
        train_Y = Variable(train_Y.cuda())
        test_feats = Variable(test_f.cuda())
        test_labels = Variable(test_yy.long().cuda())

        logprob = self.c(train_feats, test_feats, train_Y)
        junk, amax = logprob.max(1)
        acc = np.mean((amax.data.cpu().numpy().reshape(-1) == test_labels.data.cpu().numpy().reshape(-1)))
        
        _,top5 = logprob.topk(5,1)
        acc5 = 0.0
        for tp in range(5):
            acc5 += np.mean(top5[:,tp].data.cpu().numpy().reshape(-1) == test_labels.data.cpu().numpy().reshape(-1))

        return acc, acc5


    def train_on_instance(self, z, x, **kwargs):
        raise NotImplementedError()

    def sample(self, bs, seed=None):
        raise NotImplementedError()

    def prepare_batch(self, batch):
        raise NotImplementedError()

    def train(self,
              data,
              epochs,
              model_dir,
              result_dir,
              sample_num,
              save_every=1,
              val_batch_size=None,
              scheduler_fn=None,
              scheduler_args={},
              verbose=True,
             lamda = 1.0):
        for folder_name in [model_dir, result_dir]:
            if folder_name is not None and not os.path.exists(folder_name):
                os.makedirs(folder_name)
        f_mode = 'w' if not os.path.exists("%s/results.txt" % result_dir) else 'a'
        f = None
        if result_dir is not None:
            f = open("%s/results.txt" % result_dir, f_mode)
        num_classes = few_num
        train_x = data['train_data']
        train_y = data['train_label']
        train_f = data['train_feature']
        test_x = data['test_data']
        test_y = data['test_label']
        test_f = data['test_feature']
        for epoch in range(self.last_epoch, epochs):
            # Training
            epoch_start_time = time.time()
            
            train_dict = OrderedDict({'epoch': epoch+1})
            whole_classes = np.arange(num_classes)
            
            rand_labels = np.random.choice(whole_classes, num_classes, replace = False)
            num = np.random.choice(sample_num, num_classes) + 1
            
            batch_size = int(np.sum(num))
            bs = 32
            if verbose:
                pbar = tqdm(total=batch_size //bs + 1)
            train_feats = torch.zeros(batch_size, 1000)
            gan_feats = torch.zeros(batch_size, 1000)
            train_Y = torch.zeros(batch_size, num_classes)
            test_feats = torch.zeros(num_classes,1000)
            test_labels = torch.arange(num_classes)

            gan_labels = torch.arange(batch_size)

            count = 0
            # z_batches = torch.zeros(batch_size, 128)
            # gan_samples = torch.zeros(batch_size, 3, 64, 64)
            norm_c = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            gan_idx = np.zeros((batch_size,), dtype='int')
            for i in range(num_classes):
                idx = np.where(train_y==rand_labels[i])[0]
                train_idx = np.sort(np.random.choice(idx, num[i], replace = False))
                gan_idx[count:count+num[i]] += train_idx
                test_idx = np.random.choice(idx)

                gan_labels[count:count+num[i]] = i

                train_sample = torch.Tensor(train_f[list(train_idx)])
                train_feats[count: count + num[i]] = train_sample
                train_Y[count: count+num[i], i] = 1
                
                test_sample = torch.Tensor(train_f[test_idx])
                test_feats[i] = test_sample
                count += num[i]
            #############
            # Train g and d
            for i in range(batch_size // bs+1):
                if i == (batch_size // bs):
                    if i*bs == batch_size:
                        continue
                    gan_sample = torch.Tensor(train_x[list(gan_idx[i*bs:])])
                    gan_sample_f = torch.Tensor(train_f[list(gan_idx[i*bs:])])
                    gan_label = gan_labels[i*bs:]
                else:
                    gan_sample = torch.Tensor(train_x[list(gan_idx[i*bs:i*bs+bs])])
                    gan_sample_f = torch.Tensor(train_f[list(gan_idx[i*bs:i*bs+bs])])
                    gan_label = gan_labels[i*bs:i*bs + bs]
                gan_sample = (gan_sample - 0.5) / 0.5
                z_batch = self.sample_z_from_f(gan_sample_f)
                losses, outputs = self.train_on_instance(z_batch.cuda(), gan_sample.cuda(),
                                                        iter=1)

                gan_img = outputs['gz']/2.0 + 0.5
                for k in range(gan_img.size(0)):
                    gan_img[k] = norm_c(gan_img[k])
                self.fe.eval()
                gan_feat = self.fe.resnet(gan_img)

                # category loss
                logprob = self.c(train_feats.cuda(), gan_feat, train_Y.cuda())
                junk, amax = logprob.max(1)
                self.g.zero_grad()
                self.optim['g'].zero_grad()

                loss = lamda * self.c_loss(logprob, gan_label.cuda())
                loss.backward()
                self.optim['g'].step()
                ###########################################


                if i == (batch_size // bs):
                    gan_feats[i*bs:,] += gan_feat.cpu().detach()
                else:
                    gan_feats[i*bs:i*bs+bs,:] += gan_feat.cpu().detach()


                for key in losses:
                    this_key = 'train_%s' % key
                    if this_key not in train_dict:
                        train_dict[this_key] = []
                    train_dict[this_key].append(losses[key])
                pbar.update(1)
            pbar.set_postfix(self._get_stats(train_dict, 'train'))
            # Process handlers.
            for handler_fn in self.handlers:
                handler_fn(losses, z_batch, outputs,
                            {'epoch':epoch+1, 'iter':1, 'mode':'train'})
            #############

            train_feats = torch.cat([train_feats, gan_feats], axis = 0)
            train_Y = torch.cat([train_Y, train_Y], axis = 0)
        
            train_feats = Variable(train_feats.cuda())
            train_Y = Variable(train_Y.cuda())
            test_feats = Variable(test_feats.cuda())
            test_labels = Variable(test_labels.long().cuda())

            logprob = self.c(train_feats, test_feats, train_Y)
            junk, amax = logprob.max(1)
            self.c.zero_grad()
            self.optim['c'].zero_grad()

            loss = self.c_loss(logprob, test_labels)
            loss.backward()
            self.optim['c'].step()
            acc = np.mean((amax.data.cpu().numpy().reshape(-1) == test_labels.data.cpu().numpy().reshape(-1)))
            
            test_acc, test_acc_5 = self.test_network(whole_classes, num_classes, train_f, train_y, test_f, test_y)

            if verbose:
                pbar.close()
            # Step learning rates.
            for key in self.scheduler:
                self.scheduler[key].step()
            all_dict = train_dict
            all_dict['train_c_loss'] = loss.item()
            all_dict['train_c_acc'] = acc
            all_dict['test_acc'] = test_acc
            all_dict['test_acc_5'] = test_acc_5
            for key in all_dict:
                all_dict[key] = np.mean(all_dict[key])
            for key in self.optim:
                all_dict["lr_%s" % key] = \
                    self.optim[key].state_dict()['param_groups'][0]['lr']
            all_dict['time'] = \
                time.time() - epoch_start_time
            str_ = ",".join([str(all_dict[key]) for key in all_dict])
            print(str_)
            if f is not None:
                if (epoch+1) == 1:
                    # If we're not resuming, then write the header.
                    f.write(",".join(all_dict.keys()) + "\n")
                f.write(str_ + "\n")
                f.flush()
            if (epoch+1) % save_every == 0 and model_dir is not None:
                self.save(filename="%s/%i.pkl" % (model_dir, epoch+1),
                          epoch=epoch+1)

        if f is not None:
            f.close()
    
    def few(self,
              data,
              epochs,
              model_dir,
              result_dir,
              sample_num,
              save_every=1,
              val_batch_size=None,
              scheduler_fn=None,
              scheduler_args={},
              verbose=True,
              lamda = 1.0):
        for folder_name in [model_dir, result_dir]:
            if folder_name is not None and not os.path.exists(folder_name):
                os.makedirs(folder_name)
        f_mode = 'w' if not os.path.exists("%s/results.txt" % result_dir) else 'a'
        f = None
        if result_dir is not None:
            f = open("%s/results.txt" % result_dir, f_mode)
        num_classes = self.num_classes - few_num
        train_x = data['train_data']
        train_y = data['train_label']
        train_f = data['train_feature']
        test_x = data['test_data']
        test_y = data['test_label']
        test_f = data['test_feature']
        for epoch in range(self.last_epoch, epochs):
            # Training
            epoch_start_time = time.time()
            
            train_dict = OrderedDict({'epoch': epoch+1})
            whole_classes = np.arange(few_num,self.num_classes)
            
            rand_labels = np.random.choice(whole_classes, num_classes, replace = False)
            num = np.random.choice(sample_num, num_classes) + 1
            
            batch_size = int(np.sum(num))
            bs = 32
            if verbose:
                pbar = tqdm(total=batch_size //bs + 1)
            train_feats = torch.zeros(batch_size, 1000)
            gan_feats = torch.zeros(batch_size, 1000)
            train_Y = torch.zeros(batch_size, num_classes)
            test_feats = torch.zeros(num_classes,1000)
            test_labels = torch.arange(num_classes)

            gan_labels = torch.arange(batch_size)


            count = 0
            # z_batches = torch.zeros(batch_size, 128)
            # gan_samples = torch.zeros(batch_size, 3, 64, 64)
            norm_c = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            gan_idx = np.zeros((batch_size,), dtype='int')
            for i in range(num_classes):
                idx = np.where(train_y==rand_labels[i])[0][:few_k]
                train_idx = np.sort(np.random.choice(idx, num[i], replace = False))
                gan_idx[count:count+num[i]] += train_idx
                test_idx = np.random.choice(idx)

                gan_labels[count:count+num[i]] = i

                train_sample = torch.Tensor(train_f[list(train_idx)])
                train_feats[count: count + num[i]] = train_sample
                train_Y[count: count+num[i], i] = 1
                
                test_sample = torch.Tensor(train_f[test_idx])
                test_feats[i] = test_sample
                count += num[i]
            #############
            # Train g and d
            for i in range(batch_size // bs+1):
                if i == (batch_size // bs):
                    if i*bs == batch_size:
                        continue
                    gan_sample = torch.Tensor(train_x[list(gan_idx[i*bs:])])
                    gan_sample_f = torch.Tensor(train_f[list(gan_idx[i*bs:])])
                    gan_label = gan_labels[i*bs:]
                else:
                    gan_sample = torch.Tensor(train_x[list(gan_idx[i*bs:i*bs+bs])])
                    gan_sample_f = torch.Tensor(train_f[list(gan_idx[i*bs:i*bs+bs])])
                    gan_label = gan_labels[i*bs:i*bs+bs]
                gan_sample = (gan_sample - 0.5) / 0.5
                z_batch = self.sample_z_from_f(gan_sample_f)
                losses, outputs = self.train_on_instance(z_batch.cuda(), gan_sample.cuda(),
                                                        iter=1)

                gan_img = outputs['gz']/2.0 + 0.5
                for k in range(gan_img.size(0)):
                    gan_img[k] = norm_c(gan_img[k])
                self.fe.eval()
                gan_feat = self.fe.resnet(gan_img)
                if i == (batch_size // bs):
                    gan_feats[i*bs:,] += gan_feat.cpu().detach()
                else:
                    gan_feats[i*bs:i*bs+bs,:] += gan_feat.cpu().detach()

                 # category loss
                logprob = self.c(train_feats.cuda(), gan_feat, train_Y.cuda())
                junk, amax = logprob.max(1)
                self.g.zero_grad()
                self.optim['g'].zero_grad()

                loss = lamda * self.c_loss(logprob, gan_label.cuda())
#                 loss = self.c_loss(logprob, gan_label.cuda())
                loss.backward()
                self.optim['g'].step()
                ###########################################


                for key in losses:
                    this_key = 'train_%s' % key
                    if this_key not in train_dict:
                        train_dict[this_key] = []
                    train_dict[this_key].append(losses[key])
                pbar.update(1)
            pbar.set_postfix(self._get_stats(train_dict, 'train'))
            # Process handlers.
            for handler_fn in self.handlers:
                handler_fn(losses, z_batch, outputs,
                            {'epoch':epoch+1, 'iter':1, 'mode':'train'})
            #############

            train_feats = torch.cat([train_feats, gan_feats], axis = 0)
            train_Y = torch.cat([train_Y, train_Y], axis = 0)
        
            train_feats = Variable(train_feats.cuda())
            train_Y = Variable(train_Y.cuda())
            test_feats = Variable(test_feats.cuda())
            test_labels = Variable(test_labels.long().cuda())

            logprob = self.c(train_feats, test_feats, train_Y)
            junk, amax = logprob.max(1)
            self.c.zero_grad()
            self.optim['c'].zero_grad()

            loss = self.c_loss(logprob, test_labels)
            loss.backward()
            self.optim['c'].step()
            acc = np.mean((amax.data.cpu().numpy().reshape(-1) == test_labels.data.cpu().numpy().reshape(-1)))
            
            test_acc, test_acc_5 = self.few_test( train_f, train_y, test_f, test_y)

            if verbose:
                pbar.close()
            # Step learning rates.
            for key in self.scheduler:
                self.scheduler[key].step()
            all_dict = train_dict
            all_dict['train_c_loss'] = loss.item()
            all_dict['train_c_acc'] = acc
            all_dict['test_acc'] = test_acc
            all_dict['test_acc_5'] = test_acc_5
            for key in all_dict:
                all_dict[key] = np.mean(all_dict[key])
            for key in self.optim:
                all_dict["lr_%s" % key] = \
                    self.optim[key].state_dict()['param_groups'][0]['lr']
            all_dict['time'] = \
                time.time() - epoch_start_time
            str_ = ",".join([str(all_dict[key]) for key in all_dict])
            print(str_)
            if f is not None:
                if (epoch+1) == 1:
                    # If we're not resuming, then write the header.
                    f.write(",".join(all_dict.keys()) + "\n")
                f.write(str_ + "\n")
                f.flush()
            if (epoch+1) % save_every == 0 and model_dir is not None:
                self.save(filename="%s/%i.pkl" % (model_dir, epoch+1),
                          epoch=epoch+1)

        if f is not None:
            f.close()


    def save(self, filename, epoch, legacy=False):
        dd = {}
        dd['g'] = self.g.state_dict()
        dd['d'] = self.d.state_dict()
        dd['c'] = self.c.state_dict()
        for key in self.optim:
            dd['optim_' + key] = self.optim[key].state_dict()
        dd['epoch'] = epoch
        torch.save(dd, filename)

    def load(self, filename, legacy=False, ignore_d=False):
        """
        ignore_d: if `True`, then don't load in the
          discriminator.
        """
        if not self.use_cuda:
            map_location = lambda storage, loc: storage
        else:
            map_location = None

        dd = torch.load(filename,
                        map_location=map_location)
        self.g.load_state_dict(dd['g'])
        if not ignore_d:
            self.d.load_state_dict(dd['d'])
            self.c.load_state_dict(dd['c'])
        for key in self.optim:
            if ignore_d:
                if key == 'd':
                    continue
                if key == 'c':
                    continue
            self.optim[key].load_state_dict(dd['optim_'+key])
        self.last_epoch = dd['epoch']