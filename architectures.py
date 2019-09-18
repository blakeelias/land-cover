'''Heavily modified from https://github.com/pietz/unet-keras'''
import keras.backend as K
from keras.models import Input, Model
from keras.layers import Conv2D, Concatenate, MaxPooling2D, Conv2DTranspose, Activation
from keras.layers import UpSampling2D, Dropout, BatchNormalization

from pdb import set_trace as b

#------------------
# TODO: implement different parameters in the unet and dense blocks
#------------------

def conv_bn_relu(m, dim):
    x = Conv2D(dim, 3, activation=None, padding='same')(m)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def unet_block(m, dim, res=False):
    x = m
    for i in range(3):
        x = conv_bn_relu(x, dim)
    return Concatenate()([m, x]) if res else x

def dense_block(m, dim):
    x = m
    outputs = [x]
    for i in range(3):
        conv = conv_bn_relu(x, dim)
        x = Concatenate()([conv, x])
        outputs.append(conv)
    return Concatenate()(outputs)

def level_block_fixed_dims(m, dims, depth, acti, do, bn, mp, up, res, dense=False):
    max_depth = len(dims)-1
    dim = dims[max_depth-depth]
    if depth > 0:
        n = dense_block(m, dim) if dense else unet_block(m, dim, res)
        m = MaxPooling2D()(n) if mp else Conv2D(dim, 3, strides=2, padding='same')(n)
        m = level_block_fixed_dims(m, dims, depth-1, acti, do, bn, mp, up, res)
        if up:
            m = UpSampling2D()(m)
            m = Conv2D(dim, 2, activation=acti, padding='same')(m)
        else:
            m = Conv2DTranspose(dim, 3, strides=2, activation=acti, padding='same')(m)
        n = Concatenate()([n, m])
        m = dense_block(n, dim) if dense else unet_block(n, dim, res)
    else:
        m = dense_block(m, dim) if dense else unet_block(m, dim, res)
    return m

def UNet(img_shape, dims=[32, 64, 128, 256, 128], out_ch=1, activation='relu', dropout=False, batchnorm=True, maxpool=True, upconv=True, residual=False):
	i = Input(shape=img_shape)
	o = level_block_fixed_dims(i, dims, len(dims)-1, activation, dropout, batchnorm, maxpool, upconv, residual, dense=False)
	o = Conv2D(out_ch, 1, activation=None, name="logits")(o)
	return i, o

def local_avg(self, data, radius, stride=1):
    w,h = data.shape[-2:]
    diameter = 2 * radius + 1
    new_data = data.reshape(-1, 1, w, h)
    normalization_data = np.ones(new_data.shape)
    mean = keras.layers.AveragePooling2D(new_data, pool_size=(diameter, diameter), strides=(stride, stride), padding='same')
    normalization = keras.layers.AveragePooling2D(normalization_data, pool_size=(diameter, diameter), strides=(stride, stride), padding='same')
    mean_normalized = mean / normalization
    
    if stride>1:
        raise Exception('stride > 1 not implemented yet - convert Torch code to Keras')
    # Torch version:
    # mean_normalized = torch.nn.functional.interpolate(mean_normalized, size=(w,h), mode='bilinear')

    return mean_normalized.view(data.shape)
    
    
def ClusterVoting(pred_shape, num_clusters, radius):
    width, height, num_labels = pred_shape
    preds = Input(shape=pred_shape)
    clusters_shape = (width, height, num_clusters)
    clusters = Input(shape=clusters_shape)

    b()
    norms = local_avg(clusters.view((-1, width, height), radius, 1).view(clusters_shape))
    pairs = tf.einsum('bcxy, bzxy -> bczxy', clusters, preds)
    votes = local_avg(pairs.contiguous().view((-1, width, height)), radius, 1).view(pairs.shape)

    votes = (votes + 0.0001) / (norms.unsqueeze(2) + 0.01)

    output = (votes * clusters.unsqueeze(2)).sum(1) # c
    output = output[:,:-1,:,:]
    output /= output.sum(1).unsqueeze(1) # z
    
    return [pred, clusters], o


def FC_DenseNet(img_shape, dims=[32, 16, 16, 16, 16], out_ch=1, activation='relu', dropout=False, batchnorm=True, maxpool=True, upconv=True, residual=False):
    i = Input(shape=img_shape)
    o = level_block_fixed_dims(i, dims, len(dims)-1, activation, dropout, batchnorm, maxpool, upconv, residual, dense=True)
    o = Conv2D(out_ch, 1, activation=None, name="logits")(o)
    return i, o

def FCN_Small(img_shape, out_ch):
    i = Input(shape=img_shape)
    #x = Conv2D(256, (3,3), activation=LeakyReLU(alpha=0.3), padding='same')
    x = conv_bn_relu(i, 64)
    for j in range(15):
        x = conv_bn_relu(x, 64)
    o = Conv2D(out_ch, 1, activation=None, name="logits")(x)
    return i, o

if __name__ == "__main__":
    K.clear_session()
    i,o = FC_DenseNet((240,240,4), dims=[32, 16, 16, 16, 16], out_ch=5)
    model = Model(inputs=i, outputs=o)
    print(model.count_params())

    K.clear_session()
    i,o = UNet((240,240,4), dims=[32, 64, 128, 256, 128], out_ch=5)
    model = Model(inputs=i, outputs=o)
    print(model.count_params())

    K.clear_session()
    i,o = UNet((240,240,4), dims=[64, 32, 32, 32, 32], out_ch=5)
    model = Model(inputs=i, outputs=o)
    print(model.count_params())
