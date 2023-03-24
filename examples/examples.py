import argparse
import os
import sys
from pprint import pprint
import shlex

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.backends import cudnn

import sigprop
from sigprop import functional as sigpropF
from sigprop.utils import shape_numel

##############################
#   TOC:
#       STARTUP
#       CLASSES
#       HELPER METHODS
#       TRAINING METHODS
##############################

##############################
#   STARTUP
##############################

parser = argparse.ArgumentParser(description='Examples of Running Sigprop')

parser.add_argument('--dataset', default='CIFAR10',
                    help='MNIST, FashionMNIST, CIFAR10, CIFAR100, SVHN, STL10')
parser.add_argument('--batch-size', type=int, default=128,
                    help='input batch size for training')

parser.add_argument('--model', default='vgg8',
                    help='vgg8, vgg11')
parser.add_argument('--num-layers', type=int, default=1,
                    help='number of hidden fully-connected layers')
parser.add_argument('--num-hidden', type=int, default=1024,
                    help='number of hidden units for fully-connected layers')
parser.add_argument('--feat-mult', type=float, default=1,
                    help='number to multiply CNN features by')
parser.add_argument('--nonlin', default='leakyrelu',
                    help='relu or leakyrelu')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout after each nonlinearity')
parser.add_argument('--norm', default="batch_norm",
                    help='none,batch_norm,instance_norm')
parser.add_argument('--pre-act', action='store_true', default=False,
                    help='use activations (including norm) before layer')
parser.add_argument('--loss-version', type=str, default="ce",
                    help='ce')
parser.add_argument('--no-bias', action='store_true', default=False,
                    help='remove biases')

parser.add_argument('--epochs', type=int, default=400,
                    help='number of training epochs')
parser.add_argument('--lr', type=float, default=5e-4,
                    help='learning rate')
parser.add_argument('--lr-decay-milestones', nargs='+', type=int, default=[200,300,350,375],
                    help='epoch milestones to decay learning rate at')
parser.add_argument('--lr-decay-fact', type=float, default=0.25,
                    help='decay learning rate by this factor at milestone epochs')
parser.add_argument('--optim', default='adam',
                    help='adam, amsgrad or sgd')
parser.add_argument('--momentum', type=float, default=0.0,
                    help='sgd momentum')
parser.add_argument('--weight-decay', type=float, default=0.0,
                    help='weight decay')

parser.add_argument('--seed', type=int, default=1,
                    help='set random seed')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disable CUDA')

parser.add_argument('--data-dir', default='./data', type=str,
                    help='the directory to load data from')
parser.add_argument('--save-dir', default='', type=str,
                    help='the directory to save logs and models')
parser.add_argument('--resume', default='', type=str,
                    help='the checkpoint file to load the model from')

parser.add_argument('--eval-freq', type=int, default=1,
                    help='eval frequency')
parser.add_argument('--save-freq', type=int, default=-1,
                    help='save frequency')

parser.add_argument('--cutout', action='store_true', default=False,
                    help='enable cutout regularization')
parser.add_argument('--n-patches', type=int, default=1,
                    help='number of patches to cutout from the input')
parser.add_argument('--length', type=int, default=16,
                    help='the pixel length of the cutout patches')

parser.add_argument('--sigprop', action='store_true', default=False,
                    help='enable sigprop')
parser.add_argument('--sp-fine', action='store_true', default=False,
                    help='enable sigprop in blocks')

print(sys.argv)
print(shlex.join(sys.argv))
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    cudnn.enabled = True
    cudnn.benchmark = True

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

args.bias = not args.no_bias

##############################
#   CLASSES
##############################

class Cutout(object):
    def __init__(self, num_patches, length):
        self.num_patches = num_patches
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)

        mask = torch.ones((h, w))

        for n in range(self.num_patches):
            x = torch.randint(h)
            y = torch.randint(w)

            l = self.length // 2
            x1 = torch.clamp(x - l, 0, h)
            x2 = torch.clamp(x + l, 0, h)
            y1 = torch.clamp(y - l, 0, w)
            y2 = torch.clamp(y + l, 0, w)

            mask[x1:x2, y1:y2] = 0.

        mask = mask.expand_as(img)
        img = img * mask

        return img

class BlockLinear(nn.Module):
    def __init__(self, num_in, num_out, dropout=None, norm=None, sp_manager=None):
        super(BlockLinear, self).__init__()

        self.dropout_p = args.dropout if dropout is None else dropout
        self.norm = args.norm != "none" if norm is None else norm
        self.encoder = nn.Linear(num_in, num_out, bias=args.bias)
        if sp_manager is not None:
            self.encoder = sp_manager.add_propagator(self.encoder)

        if self.norm:
            self.bn = norm_1d(num_out, True)
            if sp_manager is not None:
                self.bn = sp_manager.add_propagator(self.bn)

        if args.nonlin == 'relu':
            self.nonlin = nn.ReLU(inplace=True)
        elif args.nonlin == 'leakyrelu':
            self.nonlin = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        if sp_manager is not None:
            self.nonlin = sp_manager.add_propagator(self.nonlin, sigprop.propagators.Fixed)

        if self.dropout_p > 0:
            self.dropout = torch.nn.Dropout(p=self.dropout_p, inplace=False)
            if sp_manager is not None:
                self.dropout = sp_manager.add_propagator(self.dropout, sigprop.propagators.Fixed)

    def forward(self, x):
        h = self.encoder(x)

        if self.norm:
            h = self.bn(h)
        h = self.nonlin(h)

        h_return = h
        if self.dropout_p > 0:
            h_return = self.dropout(h_return)

        return h_return

class BlockConv(nn.Module):
    def __init__(self,
            ch_in, ch_out, kernel_size, stride, padding,
            first_layer=False, dropout=None, bias=None, act="post",
            groups=1, sp_manager=None):
        super(BlockConv, self).__init__()

        self.ch_in = ch_in
        self.ch_out = ch_out
        self.dropout_p = args.dropout if dropout is None else dropout
        self.bias = True if bias is None else bias
        if not args.bias:
            self.bias = False
        self.norm = args.norm != "none"
        if act == "post":
            self.post_act = True
            self.pre_act = False
        elif act == "pre":
            self.post_act = False
            self.pre_act = True
        elif act == "cfg":
            self.pre_act = args.pre_act
            self.post_act = not self.pre_act
        elif act == "cfg_input":
            self.pre_act = False
            self.post_act = not self.pre_act
        elif act == "norm":
            self.post_act = False
            self.pre_act = False
        else:
            self.post_act = False
            self.pre_act = False
            self.norm = False
        self.encoder = nn.Conv2d(
            ch_in, ch_out, kernel_size,
            stride=stride, padding=padding, bias=self.bias,
            groups=groups,
        )
        if sp_manager is not None:
            self.encoder = sp_manager.add_propagator(self.encoder)

        if args.norm != "none" and self.norm:
            if self.pre_act:
                self.bn = norm_2d(ch_in, False)
            elif self.post_act:
                self.bn = norm_2d(ch_out, True)
            else:
                self.bn = norm_2d(ch_out, False)
            if sp_manager is not None:
                self.bn = sp_manager.add_propagator(self.bn)

        if args.nonlin == 'relu':
            self.nonlin = nn.ReLU(inplace=True)
        elif args.nonlin == 'leakyrelu':
            self.nonlin = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        if sp_manager is not None:
            self.nonlin = sp_manager.add_propagator(self.nonlin, sigprop.propagators.Fixed)

        if self.dropout_p > 0:
            self.dropout = torch.nn.Dropout2d(p=self.dropout_p, inplace=False)
            if sp_manager is not None:
                self.dropout = sp_manager.add_propagator(self.dropout, sigprop.propagators.Fixed)

    def forward(self, x, x_shortcut=None):

        if self.pre_act:
            if self.norm:
                x = self.bn(x)
            x = self.nonlin(x)
            if self.dropout_p > 0:
                x = self.dropout(x)
            h = self.encoder(x)
            if x_shortcut is not None:
                h = h + x_shortcut

        elif self.post_act:
            h = self.encoder(x)
            if self.norm:
                h = self.bn(h)
            if x_shortcut is not None:
                h = h + x_shortcut
            h = self.nonlin(h)
            if self.dropout_p > 0:
                h = self.dropout(h)

        else:
            h = self.encoder(x)
            if self.norm:
                h = self.bn(h)
            if x_shortcut is not None:
                h = h + x_shortcut

        return h

class Net(nn.Module):
    def __init__(self, num_layers, num_hidden, input_dim, input_ch, num_classes, sp_manager):
        super(Net, self).__init__()

        self.sp_manager = sp_manager
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        if not args.sp_fine:
            sp_manager = None
        self.layers = nn.ModuleList([BlockLinear(input_dim*input_dim*input_ch, num_hidden, sp_manager=sp_manager)])
        self.layers.extend([
            BlockLinear(int(num_hidden),int(num_hidden),sp_manager=sp_manager)\
            for i in range(1, num_layers)
        ])
        self.layer_out = nn.Linear(int(num_hidden), num_classes, bias=args.bias)

        if self.sp_manager is not None:

            if not args.sp_fine:
                layers_sp = []
                for l in self.layers:
                    layers_sp.append( self.sp_manager.add_propagator(l) )
                self.layers = nn.ModuleList(layers_sp)

            self.layer_out = self.sp_manager.add_propagator(self.layer_out, sigprop.propagators.Identity)

    def forward(self, x):
        if self.sp_manager:
            x = sigpropF.propagators.forward(x, lambda x : x.view(x.size(0), -1))
        else:
            x = x.view(x.size(0), -1)
        for i, layer in enumerate(self.layers):
            x = layer(x)
        x = self.layer_out(x)
        return x

class VGGn(nn.Module):
    cfgs = {
        'vgg6':  [128, 'M', 256, 'M', 512, 'M', 512, 'M'],
        'vgg8':  [128, 256, 'M', 256, 512, 'M', 512, 'M', 512, 'M'],
        'vgg11': [128, 128, 128, 256, 'M', 256, 512, 'M', 512, 512, 'M', 512, 'M'],
    }

    def __init__(self, vgg_name, input_dim, input_ch, num_classes, feat_mult=1, sp_manager=None):
        super(VGGn, self).__init__()

        self.sp_manager = sp_manager
        self.cfg = self.cfgs[vgg_name]
        self.input_dim = input_dim
        self.input_ch = input_ch
        self.num_classes = num_classes

        if not args.sp_fine:
            sp_manager = None
        self.features, self.output_dim =\
            self._make_layers(
                self.cfg, num_classes, input_ch, input_dim, feat_mult,
                sp_manager
            )

        for layer in self.cfg:
            if isinstance(layer, int):
                output_ch = layer
        self.output_ch = int(output_ch * feat_mult)

        if args.num_layers > 0:
            self.classifier = Net(args.num_layers, args.num_hidden, self.output_dim, self.output_ch, num_classes, self.sp_manager)
        else:
            self.classifier = nn.Linear(self.output_dim*self.output_dim*self.output_ch, num_classes, bias=args.bias)

            if self.sp_manager is not None:
                self.classifier = self.sp_manager.add_propagator(self.classifier, sigprop.propagators.Identity)

    def forward_body(self, x):
        for i,layer in enumerate(self.cfg):
            x = self.features[i](x)
        return x

    def forward_head(self, x):
        if args.num_layers > 0:
            x = self.classifier(x)
        else:
            if self.sp_manager is not None:
                x = sigprop.propagators.functional.forward(x, lambda x: x.view(x.size(0), -1))
            else:
                x = x.view(x.size(0), -1)
            x = self.classifier(x)
        return x

    def forward(self, x):
        x = self.forward_body(x)
        x = self.forward_head(x)
        return x

    def _make_layers(self, cfg, num_classes, input_ch, input_dim, feat_mult, sp_manager):
        layers = []
        first_layer = True
        scaler = 1
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                scaler *=2
            elif x == 'A':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
                scaler *=2
            else:
                x = int(x * feat_mult)
                if first_layer and input_dim > 64:
                    scaler = 2
                    layers += [BlockConv(input_ch, x, kernel_size=7, stride=2, padding=3, sp_manager=sp_manager)]
                else:
                    layers += [BlockConv(input_ch, x, kernel_size=3, stride=1, padding=1, sp_manager=sp_manager)]
                input_ch = x
                first_layer = False

            if self.sp_manager is not None:
                if isinstance(x, int):
                    if not args.sp_fine:
                        layers[-1] = self.sp_manager.add_propagator(layers[-1])
                else:
                    layers[-1] = self.sp_manager.add_propagator(layers[-1], sigprop.propagators.Fixed)

        return nn.Sequential(*layers), input_dim//scaler

class ModelWrapper(nn.Module):
    def __init__(self,
            model, dataset_info
        ):
        super().__init__()

        self.model = model

        if args.cuda:
            self.model.cuda()

        self.optimizer = OptimizerWrapper(
            model
        )

    def parameters(self):
        return self.model.parameters()

    def forward(self, x, y_onehot=None):
        if args.sigprop:
            return self.model((x,y_onehot))
        else:
            return self.model(x)

class HyperParamsWrapper(object):
    def __init__(self,
            optimizer
        ):
        super().__init__()
        self.optimizer = optimizer
        self.build_hyper_params()

    def build_hyper_params(self):
        self.lr = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            args.lr_decay_milestones,
            args.lr_decay_fact
        )

    def update(self, epoch=None):
        self.lr.step(epoch)

class OptimizerWrapper(object):
    def __init__(self,
            model
        ):
        super().__init__()

        self.model = model
        self.build_optimizer()
        self.hyper_params = HyperParamsWrapper(
            self.optimizer
        )

    def build_optimizer(self):
        self.optimizer = build_optimizer(self.model)

class SaveModel(object):
    def __init__(self):
        self.prepare()

    def prepare(self):
        if args.save_dir != '':
            dirname = os.path.join(args.save_dir, args.dataset)
            dirname = os.path.join(dirname, '{}'.format(args.model))

            if not os.path.exists(dirname):
                os.makedirs(dirname)
            elif os.path.exists(dirname):
                for f in os.listdir(dirname):
                    os.remove(os.path.join(dirname, f))

            self.dirname = dirname

        else:
            self.dirname = None

    def save(self, epoch, model, optimizer, info):
        if args.save_dir != '':
            torch.save({
                'epoch': epoch,
                'args': args,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'info': info,
            }, os.path.join(self.dirname, 'chkp_last_epoch.tar'))

    @staticmethod
    def load():
        checkpoint = None
        if len(args.resume) > 0:
            if os.path.isfile(args.resume):
                checkpoint = torch.load(args.resume)

                non_default = {
                    opt.dest: getattr(args, opt.dest)
                    for opt in parser._option_string_actions.values()
                    if hasattr(args, opt.dest) and opt.default != getattr(args, opt.dest)
                }
                for key, value in vars(checkpoint['args']).items():
                    setattr(args, key, value)
                for key, value in non_default.items():
                    setattr(args, key, value)
                print('[Info][Checkpoint Loaded "{}"]\t[epoch {}]'.format(args.resume, checkpoint['epoch']))

            else:
                raise RuntimeError()

        return checkpoint

##############################
#   HELPER METHODS
##############################

def build_dataset():

    data_dir = args.data_dir

    if args.dataset == 'MNIST':
        dataset_info = dict(
            input_dim = 28,
            input_ch = 1,
            num_classes = 10,
        )
        train_transform = transforms.Compose([
                transforms.RandomCrop(28, padding=2),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        if args.cutout:
            train_transform.transforms.append(Cutout(num_patches=args.n_patches, length=args.length))
        dataset_train = datasets.MNIST(data_dir+'/MNIST', train=True, download=True, transform=train_transform)
        dataset_test = datasets.MNIST(
            data_dir+'/MNIST', train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        )

    elif args.dataset == 'FashionMNIST':
        dataset_info = dict(
            input_dim = 28,
            input_ch = 1,
            num_classes = 10,
        )
        train_transform = transforms.Compose([
                transforms.RandomCrop(28, padding=2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.286,), (0.353,))
            ])
        if args.cutout:
            train_transform.transforms.append(Cutout(num_patches=args.n_patches, length=args.length))
        dataset_train = datasets.FashionMNIST(data_dir+'/FashionMNIST', train=True, download=True, transform=train_transform)
        dataset_test = datasets.FashionMNIST(
            data_dir+'/FashionMNIST', train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.286,), (0.353,))
            ])
        )

    elif args.dataset == 'CIFAR10':
        dataset_info = dict(
            input_dim = 32,
            input_ch = 3,
            num_classes = 10,
        )
        train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.424, 0.415, 0.384), (0.283, 0.278, 0.284))
            ])
        if args.cutout:
            train_transform.transforms.append(Cutout(num_patches=args.n_patches, length=args.length))
        dataset_train = datasets.CIFAR10(data_dir+'/CIFAR10', train=True, download=True, transform=train_transform)
        dataset_test = datasets.CIFAR10(
            data_dir+'/CIFAR10', train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.424, 0.415, 0.384), (0.283, 0.278, 0.284))
            ])
        )

    elif args.dataset == 'CIFAR100':
        dataset_info = dict(
            input_dim = 32,
            input_ch = 3,
            num_classes = 100,
        )
        train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.438, 0.418, 0.377), (0.300, 0.287, 0.294))
            ])
        if args.cutout:
            train_transform.transforms.append(Cutout(num_patches=args.n_patches, length=args.length))
        dataset_train = datasets.CIFAR100(data_dir+'/CIFAR100', train=True, download=True, transform=train_transform)
        dataset_test = datasets.CIFAR100(data_dir+'/CIFAR100', train=False,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.438, 0.418, 0.377), (0.300, 0.287, 0.294))
                ]))

    elif args.dataset == 'SVHN':
        dataset_info = dict(
            input_dim = 32,
            input_ch = 3,
            num_classes = 10,
        )
        train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.431, 0.430, 0.446), (0.197, 0.198, 0.199))
            ])
        if args.cutout:
            train_transform.transforms.append(Cutout(num_patches=args.n_patches, length=args.length))
        dataset_train = torch.utils.data.ConcatDataset((
            datasets.SVHN(data_dir+'/SVHN', split='train', download=True, transform=train_transform),
            datasets.SVHN(data_dir+'/SVHN', split='extra', download=True, transform=train_transform)))
        dataset_test = datasets.SVHN(data_dir+'/SVHN', split='test', download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.431, 0.430, 0.446), (0.197, 0.198, 0.199))
                ]))

    elif args.dataset == 'STL10':
        dataset_info = dict(
            input_dim = 96,
            input_ch = 3,
            num_classes = 10,
        )
        train_transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.447, 0.440, 0.407), (0.260, 0.257, 0.271))
            ])
        if args.cutout:
            train_transform.transforms.append(Cutout(num_patches=args.n_patches, length=args.length))
        dataset_train = datasets.STL10(data_dir+'/STL10', split='train', download=True, transform=train_transform)
        dataset_test = datasets.STL10(data_dir+'/STL10', split='test',
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.447, 0.440, 0.407), (0.260, 0.257, 0.271))
                ]))

    else:
        raise RuntimeError('Not a valid dataset')

    dataset_info = argparse.Namespace(**dataset_info)

    return dataset_info, dataset_train, dataset_test

def build_dataloader(dataset_train, dataset_test):
    kwargs_train = dict(shuffle = True)
    kwargs_test = dict(shuffle = False)

    if args.cuda:
        kwargs_train.update(dict(num_workers = 0, pin_memory = True))
        kwargs_test.update(dict(num_workers = 0, pin_memory = True))

    if dataset_train is not None:
        train_loader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=args.batch_size, **kwargs_train
        )
    else:
        train_loader = None

    if dataset_test is not None:
        test_loader = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=args.batch_size, **kwargs_test
        )
    else:
        test_loader = None

    return train_loader, test_loader

def build_model(dataset_info, sp_manager, input_ch=None):

    if input_ch is None:
        input_ch = dataset_info.input_ch

    kwargs = dict(
        input_dim = dataset_info.input_dim,
        input_ch =  input_ch,
        num_classes = dataset_info.num_classes,
        sp_manager = sp_manager
    )

    model = None
    if args.model == 'mlp':
        model = Net(num_layers=args.num_layers, num_hidden=args.num_hidden, **kwargs)
    elif args.model.startswith('vgg'):
        model = VGGn(vgg_name=args.model, feat_mult=args.feat_mult, **kwargs)

    if model is None:
        print('No valid model defined')

    if sp_manager is not None:
        model = sp_manager.set_model(model)

    return model

def build_optimizer(model, lr=None, weight_decay=None):
    if lr is None:
        lr = args.lr
    if weight_decay is None:
        weight_decay = args.weight_decay
    p = list(model.parameters())
    if len(p) == 0:
        return None
    if args.optim == 'sgd':
        optimizer = optim.SGD(
            p, lr=lr,
            weight_decay=weight_decay,
            momentum=args.momentum)
    elif args.optim == 'adam' or args.optim == 'amsgrad':
        optimizer = optim.Adam(
            p, lr=lr,
            weight_decay=weight_decay,
            #betas=args.betas,
            amsgrad=args.optim == 'amsgrad')
    else:
        raise RuntimeError('Unknown optimizer')

    return optimizer

def signal_modules_fp(input_shape, output_shape, bias=False):
    return nn.Linear(
        int(shape_numel(input_shape)),
        int(shape_numel(output_shape)),
        bias=bias
    )

def signal_modules_lp(input_shape, output_shape, bias=False):
    return nn.Sequential(
        nn.Linear(
            int(shape_numel(input_shape)),
            int(shape_numel(output_shape)),
            bias=bias
        ),
        nn.LayerNorm(shape_numel(output_shape)),
        nn.ReLU()
        #nn.Tanh()
    )

def signal_modules_lpi(input_shape, output_shape, bias=False):
    lpi_signal_module = nn.Sequential(
        nn.Linear(
            int(shape_numel(input_shape)),
            int(shape_numel(output_shape)),
            bias=bias
        ),
        nn.LayerNorm(shape_numel(output_shape)),
        nn.ReLU()
    )
    lpi_input_module = nn.Sequential(
        nn.Conv2d(
            3, output_shape[0], 3, 1, 1,
            bias=bias
        ),
        nn.BatchNorm2d(output_shape[0]),
        nn.ReLU()
    )
    return lpi_signal_module, lpi_input_module

def count_parameters(model):
    ''' Count number of parameters in model influenced by global loss. '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calc_loss(output, target):
    version = args.loss_version
    if version == "ce":
        loss = F.cross_entropy(output, target)
    else:
        raise RuntimeError("Not a valid loss")

    return loss

def norm_1d(ch, fix_affine, **kwargs):
    if args.norm == "batch_norm":
        bn = torch.nn.BatchNorm1d(ch, track_running_stats=False, **kwargs)
        if fix_affine:
            nn.init.constant_(bn.weight, 1)
            nn.init.constant_(bn.bias, 0)
    elif args.norm == "instance_norm":
        bn = torch.nn.LayerNorm(
            ch,
            **kwargs
        )
        if fix_affine:
            nn.init.constant_(bn.weight, 1)
            nn.init.constant_(bn.bias, 0)
    elif args.norm == "instance_batch_norm":
        bn = torch.nn.InstanceNorm1d(
            ch,
            track_running_stats=True,
            affine=True,
            **kwargs
        )
        if fix_affine:
            nn.init.constant_(bn.weight, 1)
            nn.init.constant_(bn.bias, 0)
    else:
        raise RuntimeError("Not a valid norm")

    return bn

def norm_2d(ch, fix_affine, **kwargs):
    if args.norm == "batch_norm":
        bn = torch.nn.BatchNorm2d(ch, track_running_stats=False, **kwargs)
        if fix_affine:
            nn.init.constant_(bn.weight, 1)
            nn.init.constant_(bn.bias, 0)
    elif args.norm == "instance_norm":
        bn = torch.nn.InstanceNorm2d(
            ch,
            track_running_stats=False,
            affine=True,
            **kwargs
        )
        if fix_affine:
            nn.init.constant_(bn.weight, 1)
            nn.init.constant_(bn.bias, 0)
    elif args.norm == "instance_batch_norm":
        bn = torch.nn.InstanceNorm2d(
            ch,
            track_running_stats=True,
            affine=True,
            **kwargs
        )
        if fix_affine:
            nn.init.constant_(bn.weight, 1)
            nn.init.constant_(bn.bias, 0)
    else:
        raise RuntimeError("Not a valid norm")

    return bn

def calc_performance(output, target, num=None):
    if num is None:
        num = output.shape[1]
    pred = output.argmax(1)
    target = target
    target_onehot = F.one_hot(target, num).float()

    acc_mask = pred.eq(target)
    acc = acc_mask.sum() / acc_mask.size(0)

    return pred, target, target_onehot, acc_mask, acc

def print_train_info(
        epoch_info,
        epoch, batch, batches
        ):
    print(
        "[Info][Train  Epoch {}/{}][Batch {}/{}]"
        "\t[loss {:.4f}]\t[acc {:.4f}]".format(
            epoch, args.epochs,
            batch, batches,
            epoch_info.loss / epoch_info.count,
            epoch_info.acc / epoch_info.count,
    ))

def print_test_info(
        test_info, epoch
        ):
    print(
        "[Info][Test   Epoch {}/{}]\t\t"
        "\t[loss {:.4f}]\t[acc {:.4f}]".format(
            epoch, args.epochs,
            test_info.loss / test_info.count,
            test_info.acc / test_info.count,
    ))

##############################
#   TRAINING METHODS
##############################

class Runner(object):

    def predict(
            self, model, data, target,
        ):

        output = model(data, target)
        with torch.no_grad():
            pred, target, target_onehot, acc_mask, acc =\
                    calc_performance(output, target)

        loss = calc_loss(output, target)

        return loss, acc, acc_mask

    def train_batch(
            self, model, data, target,
        ):

        model.train()

        optimizer = model.optimizer.optimizer

        loss, acc, acc_mask =\
            self.predict(model, data, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return acc_mask,loss

    def train_epoch(
            self, epoch, model, train_loader, num_classes,
        ):

        epoch_info = argparse.Namespace()
        epoch_info.acc = 0
        epoch_info.count = 0
        epoch_info.loss = 0

        for batch, (data, target) in enumerate(train_loader):

            if args.cuda:
                data, target = data.cuda(), target.cuda()

            data_size = data.size(0)

            acc_mask, loss = self.train_batch(
                model, data, target,
            )

            epoch_info.acc += acc_mask.sum()
            assert acc_mask.shape[0] == data.shape[0]
            epoch_info.count += data.shape[0]
            epoch_info.loss += loss * data.shape[0]

            if batch == len(train_loader) - 1:
                print_train_info(epoch_info, epoch, batch, len(train_loader))

        return epoch_info

    def train(
            self, model, num_classes,
            train_loader, test_loader,
            info
        ):

        save_model = SaveModel()

        info.train = argparse.Namespace()
        info.train.acc = 0
        info.train.count = 0
        info.train.loss = 0

        time = 0

        for epoch in range(args.epochs):

            print("\n\n\n\nEpoch Start: {}".format(epoch))

            epoch_info = self.train_epoch(
                epoch, model, train_loader, num_classes
            )

            if time % args.eval_freq == 0:
                self.test(epoch, model, test_loader, num_classes)

            info.train.acc += epoch_info.acc
            info.train.count += epoch_info.count
            info.train.loss += epoch_info.loss

            if args.save_freq > 0 and time % args.save_freq == 0:
                save_model.save(
                    taskset, model, info.optimizers.slow.optimizer,
                    info.reporters.train, info.reporters.test
                )

            time += 1

    def test(self, epoch, model, test_loader, num_classes):

        model.eval()
        test_info = argparse.Namespace()
        test_info.acc = 1
        test_info.count = 1
        test_info.loss = 1

        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()

            with torch.no_grad():
                loss, acc, acc_mask =\
                    self.predict(model, data, target)

            test_info.acc += acc_mask.sum()
            assert acc_mask.shape[0] == data.shape[0]
            test_info.count += data.shape[0]
            test_info.loss += loss * data.shape[0]

        print_test_info(test_info, epoch)

        return test_info

class RunnerSigprop(Runner):
    def __init__(self, monitor_main):
        self.monitor_main = monitor_main

    def train_epoch(
            self, epoch, model, train_loader, num_classes,
        ):
        self.monitor_main.reset()
        epoch_info = super().train_epoch(
            epoch, model, train_loader, num_classes,
        )
        print(self.monitor_main.metrics())
        return epoch_info

    def test(
            self, epoch, model, test_loader, num_classes
        ):
        self.monitor_main.reset()
        test_info = super().test(
            epoch, model, test_loader, num_classes
        )
        print(self.monitor_main.metrics())
        return test_info
