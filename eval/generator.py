from __future__ import print_function

import os
import sys
import torch
import io
import time
import numpy as np
from PIL import Image
from datetime import datetime
from torch.autograd import Variable
from miscc.config import cfg
from miscc.utils import build_super_images2
from model import RNN_ENCODER, G_NET
from miscc.config import cfg

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

class Generator:
    def __init__(self, caption_file, saveable, cuda=False, profile=False):
        # flags
        self.cuda = cuda
        self.profile = profile

        if self.profile:
            print('Initializing Generator...')
            print('cuda={}\nprofile={}'.format(self.cuda, self.profile))

        # load caption indices
        x = pickle.load(open(caption_file, 'rb'))
        self.ixtoword = x[2]
        self.wordtoix = x[3]
        del x

        # load text encoder
        self.text_encoder = RNN_ENCODER(len(self.wordtoix), nhidden=cfg.TEXT.EMBEDDING_DIM)
        state_dict = torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
        self.text_encoder.load_state_dict(state_dict)
        if self.cuda:
            self.text_encoder.cuda()
            
        self.text_encoder.eval()

        # load generative model
        self.netG = G_NET()
        state_dict = torch.load(cfg.TRAIN.NET_G, map_location=lambda storage, loc: storage)
        self.netG.load_state_dict(state_dict)
        if self.cuda:
            self.netG.cuda()
            
        self.netG.eval()

        # saveable items -> push to storage
        self.saveable = saveable

    def vectorize_caption(self, caption, copies):
        # create caption vector
        tokens = caption.split(' ')
        cap_v = []
        for t in tokens:
            t = t.strip().encode('ascii', 'ignore').decode('ascii')
            if len(t) > 0 and t in self.wordtoix:
                cap_v.append(self.wordtoix[t])

        # expected state for single generation
        captions = np.zeros((copies, len(cap_v)))
        for i in range(copies):
            captions[i,:] = np.array(cap_v)
        cap_lens = np.zeros(copies) + len(cap_v)

        return captions.astype(int), cap_lens.astype(int), len(self.wordtoix)

    def generate(self, caption, copies=2):
        # load word vector
        captions, cap_lens, n_words = self.vectorize_caption(caption, copies)

        # only one to generate
        batch_size = captions.shape[0]

        nz = cfg.GAN.Z_DIM
        captions = Variable(torch.from_numpy(captions), volatile=True)
        cap_lens = Variable(torch.from_numpy(cap_lens), volatile=True)
        noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)

        if self.cuda:
            captions = captions.cuda()
            cap_lens = cap_lens.cuda()
            noise = noise.cuda()

        #######################################################
        # (1) Extract text embeddings
        #######################################################
        hidden = self.text_encoder.init_hidden(batch_size)
        words_embs, sent_emb = self.text_encoder(captions, cap_lens, hidden)
        mask = (captions == 0)

        #######################################################
        # (2) Generate fake images
        #######################################################
        noise.data.normal_(0, 1)
        fake_imgs, attention_maps, _, _ = self.netG(noise, sent_emb, words_embs, mask)

        # G attention
        cap_lens_np = cap_lens.cpu().data.numpy()

        # prefix for partitioning images
        prefix = datetime.now().strftime('%Y/%B/%d/%H_%M_%S_%f')
        urls = []
        # only look at first one
        for j in range(batch_size):
            for k in range(len(fake_imgs)):
                im = fake_imgs[k][j].data.cpu().numpy()
                im = (im + 1.0) * 127.5
                im = im.astype(np.uint8)
                im = np.transpose(im, (1, 2, 0))

                # save using saveable
                birdy = 'bird_g{}'.format(k)
                if copies > 2:
                    item = self.saveable.save('{}/{}'.format(prefix, j), birdy, im)
                else:
                    item = self.saveable.save(prefix, birdy, im)

                urls.append(item)

            if copies == 2:
                for k in range(len(attention_maps)):
                    if len(fake_imgs) > 1:
                        im = fake_imgs[k + 1].detach().cpu()
                    else:
                        im = fake_imgs[0].detach().cpu()
                            
                    attn_maps = attention_maps[k]
                    att_sze = attn_maps.size(2)

                    img_set, sentences = \
                        build_super_images2(im[j].unsqueeze(0),
                                            captions[j].unsqueeze(0),
                                            [cap_lens_np[j]], self.ixtoword,
                                            [attn_maps[j]], att_sze)

                    if img_set is not None:
                        attnmap = 'attmaps_a{}'.format(k)
                        item = self.saveable.save(prefix, attnmap, img_set)
                        urls.append(item)
            if copies == 2:
                break

        return urls
