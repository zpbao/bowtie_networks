import glob
import sys
import os
import argparse
from hologan import HoloGAN
import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from iterators.datasets import CelebADataset
from torchvision.transforms import transforms
from architectures import arch, classification
from collections import OrderedDict
from tools import (count_params)
import torchvision

import numpy as np

from distillation import Student




use_shuriken = False
try:
    from shuriken.utils import get_hparams
    use_shuriken = True
except:
    pass

# This dictionary's keys are the ones that are used
# to auto-generate the experiment name. The values
# of those keys are tuples, the first element being
# shortened version of the key (e.g. 'dataset' -> 'ds')
# and a function which may optionally shorten the value.


# data_path = '../cub-64.npz'
# num_classes = 200

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--angles', type=str, default="[0,0,0,360,0,0]")
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=32)
    parser.add_argument('--nmf', type=int, default=32)
    parser.add_argument('--nb', type=int, default=2) # num blocks
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--z_dim', type=int, default=128)
    # parser.add_argument('--z_extra_fc', action='store_true')
    parser.add_argument('--lamb', type=float, default=0.)
    parser.add_argument('--lr_g', type=float, default=2e-4)
    parser.add_argument('--lr_d', type=float, default=2e-4)
    parser.add_argument('--beta1', type=float, default=0.)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--update_g_every', type=int, default=5)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--val_batch_size', type=int, default=8)
    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--save_images_every', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--resume', type=str, default='auto')
    parser.add_argument('--trial_id', type=str, default=None)
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--sample_num', type=int, default=1)
    parser.add_argument('--few_shot',action = 'store_true')
    parser.add_argument('--lamda',type=float, default=1.0)
    parser.add_argument('--data_path', type = str, default = None)
    parser.add_argument('--num_classes', type = int, default = 200)
    args = parser.parse_args()
    return args

args = parse_args()
args = vars(args)

print("args before shuriken:")
print(args)

if use_shuriken:
    # This only applies to me. If you're not me,
    # don't worry about this code.
    shk_args = get_hparams()
    print("shk args:", shk_args)
    # Stupid bug that I have to fix: if an arg is ''
    # then assume it's a boolean.
    for key in shk_args:
        if shk_args[key] == '':
            shk_args[key] = True
    args.update(shk_args)

if args['trial_id'] is None and 'SHK_TRIAL_ID' in os.environ:
    print("SHK_TRIAL_ID found so injecting this into `trial_id`...")
    args['trial_id'] = os.environ['SHK_TRIAL_ID']
else:
    if args['trial_id'] is None:
        print("trial_id not defined so generating random id...")
        trial_id = "".join([ random.choice(string.ascii_letters[0:26]) for j in range(5) ])
        args['trial_id'] = trial_id

if 'SHK_EXPERIMENT_ID' in os.environ:
    print("SHK_EXPERIMENT_ID found so injecting this into `name`...")
    args['name'] = os.environ['SHK_EXPERIMENT_ID']
else:
    if args['name'] is None:
        raise Exception("You must give a name to this experiment")

torch.manual_seed(args['seed'])

# This is the one from the progressive growing GAN
# code.
IMG_HEIGHT = 64 

dataset = np.load(args['data_path'])


if args['save_path'] is None:
    args['save_path'] = os.environ['RESULTS_DIR']

gen, disc = arch.get_network(args['z_dim'],
                             ngf=args['ngf'],
                             ndf=args['ndf'])
classifier = classification.ResNet_C(args['z_dim'], args['num_classes'])
f_e = Student(args['num_classes']).cuda()
f_e.load_state_dict(torch.load('./student.pkl'))

print("Generator:")
print(gen)
print(count_params(gen))
print("Disc:")
print(disc)
print(count_params(disc))
print("Classifier:")
print(classifier)
print(count_params(classifier))


angles = eval(args['angles'])
gan = HoloGAN(
    gen_fn=gen,
    disc_fn=disc,
    classifier_fn = classifier,
    feat_ext = f_e,
    z_dim=args['z_dim'],
    lamb=args['lamb'],
    angles=angles,
    opt_d_args={'lr': args['lr_d'], 'betas': (args['beta1'], args['beta2'])},
    opt_g_args={'lr': args['lr_g'], 'betas': (args['beta1'], args['beta2'])},
    update_g_every=args['update_g_every'],
    num_classes = args['num_classes'],
    handlers=[]
)

def _image_handler(gan, out_dir, batch_size=32):
    def _image_handler(losses, batch, outputs, kwargs):
        if kwargs['iter'] == 1:
            if kwargs['epoch'] % args['save_images_every'] == 0:
                gan._eval()
                mode = kwargs['mode']
                if mode == 'train':
                    # TODO: do for valid as well
                    epoch = kwargs['epoch']
                    z_batch = batch
                    z_batch = z_batch.cuda()
                    for key in ['y']:
                        rot = gan._generate_rotations(z_batch,
                                                      min_angle=gan.angles['min_angle_%s' % key],
                                                      max_angle=gan.angles['max_angle_%s' % key],
                                                      axes=[key],
                                                      num=30)
                        #padding = torch.zeros_like(rot['yaw'][0])+0.5
                        save_image( torch.cat(rot[key], dim=0),
                                    nrow=z_batch.size(0),
                                    filename="%s/rot_%s_%i.png" % (out_dir, key, epoch) )

    return _image_handler


save_path = "%s/s%i/%s" % \
    (args['save_path'], args['seed'], args['name'])
if not os.path.exists(save_path):
    os.makedirs(save_path)


expt_dir = "%s/%s" % (save_path, args['trial_id'])
if not os.path.exists(expt_dir):
    os.makedirs(expt_dir)

gan.handlers.append(_image_handler(gan, expt_dir))

print("expt_dir:", expt_dir)

if args['resume'] is not None:
    if args['resume'] == 'auto':
        # autoresume
        # List all the pkl files.
        files = glob.glob("%s/*.pkl" % expt_dir)
        # Make them absolute paths.
        files = [os.path.abspath(key) for key in files]
        if len(files) > 0:
            # Get creation time and use that.
            latest_model = max(files, key=os.path.getctime)
            print("Auto-resume mode found latest model: %s" %
                  latest_model)
            gan.load(latest_model)
    else:
        print("Loading model: %s" % args['resume'])
        gan.load(args['resume'])

if args['interactive']:

    bs = 32
    gan._eval()

    z_batch = gan.sample_z(bs, seed=None)
    if gan.use_cuda:
        z_batch = z_batch.cuda()

    import numpy as np

    # -45 to +45 deg
    rot = gan._generate_rotations(z_batch,
                                  min_angle=-2*np.pi,
                                  max_angle=2*np.pi,
                                  num=50)
    padding = torch.zeros_like(rot['yaw'][0])+0.5

    imgs = torch.cat(rot['yaw'] + \
                     [padding] + \
                     rot['pitch'] + \
                     [padding] + \
                     rot['roll'], dim=0)

    save_image( imgs,
                nrow=bs,
                filename="%s/%s/gen_z.png" % (args['save_path'],
                                              args['name']))



    #import pdb
    #pdb.set_trace()
    

else:
    if args['few_shot']:
        gan.few(data = dataset,
                epochs=args['epochs'],
                model_dir=expt_dir,
                result_dir=expt_dir,
                sample_num = args['sample_num'],
                save_every=args['save_every'],
                lamda = args['lamda'])
    else:
        gan.train(data = dataset,
                epochs=args['epochs'],
                model_dir=expt_dir,
                result_dir=expt_dir,
                sample_num = args['sample_num'],
                save_every=args['save_every'],
                 lamda = args['lamda'])