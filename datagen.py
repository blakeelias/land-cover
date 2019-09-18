from collections import defaultdict
import numpy as np
import keras.utils

from utils import nlcd_classes_to_idx

def color_aug(colors):
    n_ch = colors.shape[0]
    contra_adj = 0.05
    bright_adj = 0.05

    ch_mean = np.mean(colors, axis=(-1,-2), keepdims=True).astype(np.float32)

    contra_mul = np.random.uniform(1-contra_adj, 1+contra_adj, (n_ch,1,1)).astype(np.float32)
    bright_mul = np.random.uniform(1-bright_adj, 1+bright_adj, (n_ch,1,1)).astype(np.float32)

    colors = (colors - ch_mean) * contra_mul + ch_mean * bright_mul
    return colors

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    
    def __init__(self, patches, clusters, labels, batch_size, steps_per_epoch, input_size, output_size, num_channels, do_color_aug=False, do_superres=False, superres_only_states=[], do_cluster_vote_training=False):
        'Initialization'

        self.patches = patches
        self.clusters = clusters
        self.labels = labels
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        assert steps_per_epoch * batch_size < len(patches)

        self.input_size = input_size
        self.output_size = output_size

        self.num_channels = num_channels
        self.num_classes = 5

        self.do_color_aug = do_color_aug

        self.do_superres = do_superres
        self.do_cluster_vote_training = do_cluster_vote_training
        self.superres_only_states = superres_only_states
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.steps_per_epoch

    def __getitem__(self, index):
        'Generate one batch of data'
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        patch_fns = [self.patches[i] for i in indices]
        cluster_fns = [self.clusters[i] for i in indices]
        label_fns = [self.labels[i] for i in indices]
        
        x_batch = np.zeros((self.batch_size, self.input_size, self.input_size, self.num_channels), dtype=np.float32)
        y_hr_batch = np.zeros((self.batch_size, self.output_size, self.output_size, 5), dtype=np.float32)
        y_sr_batch = None
        if self.do_superres:
            y_sr_batch = np.zeros((self.batch_size, self.output_size, self.output_size, 22), dtype=np.float32)

        clusters_batch = np.zeros((self.batch_size, self.input_size, self.input_size, self.num_clusters), dtype=np.float32)
            
        def load_from_fn(fn):
            if fn.endswith(".npz"):
                data = np.load(fn)["arr_0"].squeeze()
                # (c h w)
            elif fn.endswith(".npy"):
                data = np.load(fn).squeeze()
            return data

        
        for i, ((patch_fn, state),
                (cluster_fn, state),
                (label_fn, state)) in enumerate(zip(patch_fns, cluster_fns, label_fns)):
            data = load_from_fn(patch_fn)
            data = np.rollaxis(data, 0, 3)
            # (h w c)
            
            #do a random crop if input_size is less than the prescribed size
            assert data.shape[0] == data.shape[1]
            data_size = data.shape[0]
            if self.input_size < data_size:
                x_idx = np.random.randint(0, data_size - self.input_size)
                y_idx = np.random.randint(0, data_size - self.input_size)
                data = data[y_idx:y_idx+self.input_size, x_idx:x_idx+self.input_size, :]

            
            # setup x
            if self.do_color_aug:
                x_batch[i] = color_aug(data[:,:,:4] / 255.0)
            else:
                x_batch[i] = data[:,:,:4] / 255.0

            # setup y_highres
            y_train_hr = load_from_fn(label_fn)
            y_train_hr[y_train_hr==15] = 0
            y_train_hr[y_train_hr==5] = 4
            y_train_hr[y_train_hr==6] = 4
            y_train_hr = keras.utils.to_categorical(y_train_hr, 5)

            if self.do_superres:
                if state in self.superres_only_states:
                    y_train_hr[:,:,0] = 0
                else:
                    y_train_hr[:,:,0] = 1
            else:
                y_train_hr[:,:,0] = 0
            y_hr_batch[i] = y_train_hr
            
            # setup y_superres
            try:
                if self.do_superres:
                    y_train_nlcd = nlcd_classes_to_idx(data[:,:,9])
                    y_train_nlcd = keras.utils.to_categorical(y_train_nlcd, 22)
                    y_sr_batch[i] = y_train_nlcd
            except:
                raise Exception('No super-resolution data available.')

            # setup clusters
            clusters = load_from_fn(cluster_fn)
            # (num_clusters height width)
            clusters = np.rollaxis(clusters, 0, 3)
            # (height width num_clusters)
            clusters_batch[i] = clusters
            
        if self.do_superres:
            outputs = {"outputs_hr": y_hr_batch, "outputs_sr": y_sr_batch}
        else:
            outputs = y_hr_batch

        if self.do_cluster_vote_training:
            inputs = {"image": x_batch.copy(), "clusters": clusters_batch.copy()}
        else:
            return x_batch.copy()

        return (inputs, outputs)
        
    def on_epoch_end(self):
        self.indices = np.arange(len(self.patches))
        np.random.shuffle(self.indices)
