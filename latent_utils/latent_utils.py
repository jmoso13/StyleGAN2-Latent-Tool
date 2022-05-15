from keras.layers import Lambda, Input, Dense, Dropout, LeakyReLU, BatchNormalization, Layer, Concatenate
from keras.models import Model
from keras.losses import mse, binary_crossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import tensorflow as tf

import numpy as np


def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def dim_decay_func(x, u, l, n):
  return int(np.rint(-np.power(x, n) * (u - l) + u))


def euclid(m):
  return -(-2 * K.dot(m, K.transpose(m)) + K.sum(K.square(m), axis=1) + K.reshape(K.sum(K.square(m), axis=1),(-1,1)))


def remove_self(d):
  return K.cast(K.greater_equal(d,0), 'float32') * (K.min(d, axis=0) - 1) + d


def means(args):
  inputs = args[0]
  remove = args[1]
  idx = K.argmax(remove)
  return (inputs + K.gather(inputs, idx))/2.0


def get_vae_dict(original_dim, h_l, l, model_name, loss_weights=[0.5, 0.5], n=2.0, dropout_rate=0.0, batch_norm=False, verbose=True, nn_vae=False):
  # Input Layer
  input_shape = (original_dim,)
  inputs = Input(shape=input_shape, name='encoder_input')
  last_layer = inputs
  if dropout_rate:
    inputs_dropout = Dropout(rate=dropout_rate)(inputs)
    last_layer = inputs_dropout
  # Hidden Layers
  h_l_dict = dict()
  for x in range(h_l):
    layer_dim = dim_decay_func((x+1.0)/(h_l+1.0), original_dim, l, n)
    layer_name = 'encoder_layer_%s'%(x+1)
    h_l_dict[layer_name] = Dense(layer_dim, name=layer_name)(last_layer)
    if batch_norm:
      h_l_dict[layer_name] = BatchNormalization(name=layer_name+'_batch_norm')(h_l_dict[layer_name])
    h_l_dict[layer_name] = LeakyReLU(name=layer_name+'_leaky_relu')(h_l_dict[layer_name])
    if dropout_rate:
      h_l_dict[layer_name] = Dropout(rate=dropout_rate, name=layer_name+'_dropout')(h_l_dict[layer_name])
    last_layer = h_l_dict[layer_name]
  # VAE Layer
  z_mean = Dense(l, name='z_mean')(last_layer)
  z_log_var = Dense(l, name='z_log_var')(last_layer)
  if nn_vae:
    dist = Lambda(euclid, output_shape=(l,), name='dist')(z_mean)
    remove = Lambda(remove_self, output_shape=(l,), name='remove_self')(dist)
    m = Lambda(means, output_shape=(l,), name='m')([z_mean, remove])
    z = Lambda(sampling, output_shape=(l,), name='z')([m, z_log_var])
  else:
    z = Lambda(sampling, output_shape=(l,), name='z')([z_mean, z_log_var])
  # Instantiate encoder model
  if nn_vae:
    encoder = Model(inputs, [z_mean, m, z_log_var, z], name='encoder')
  else:
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
  # DECODER
  last_layer = z
  for x in reversed(range(h_l)):
    layer_dim = dim_decay_func((x+1.0)/(h_l+1.0), original_dim, l, n)
    layer_name = 'decoder_layer_%s'%(x+1)
    h_l_dict[layer_name] = Dense(layer_dim, name=layer_name)(last_layer)
    if batch_norm:
      h_l_dict[layer_name] = BatchNormalization(name=layer_name+'_batch_norm')(h_l_dict[layer_name])
    h_l_dict[layer_name] = LeakyReLU(name=layer_name+'_leaky_relu')(h_l_dict[layer_name])
    if dropout_rate:
      h_l_dict[layer_name] = Dropout(rate=dropout_rate, name=layer_name+'_dropout')(h_l_dict[layer_name])
    last_layer = h_l_dict[layer_name]        
  outputs = Dense(original_dim, name='outputs')(last_layer)
  # Model to train
  vae = Model(inputs, outputs, name=model_name)
  # Defining Loss
  reconstruction_loss = mse(inputs, outputs)
  reconstruction_loss *= original_dim
  if nn_vae:
    kl_loss = 1 + z_log_var - K.log(K.mean(K.square(K.max(remove, axis=0)))) - K.exp(z_log_var - K.log(K.mean(K.square(K.max(remove, axis=0)))))
  else:
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
  kl_loss = K.sum(kl_loss, axis=-1)
  kl_loss *= -0.5
  vae_loss = K.mean(loss_weights[0]*reconstruction_loss + loss_weights[1]*kl_loss)
  get_losses = K.function([inputs], [vae_loss, reconstruction_loss, kl_loss])
  get_rls = K.function([inputs], [reconstruction_loss])
  vae.add_loss(vae_loss)
  vae.compile(optimizer='adam')
  if verbose:
    print(encoder.summary())
    print(vae.summary())

  return {'vae': vae, 
          'encoder': encoder
          }

       
def build_decoder(vae, decoder_input):
  layer_list = list()
  for l in reversed(vae.layers):
    if l.name == 'z':
      break
    layer_list.append(l)
  decoder_output = decoder_input
  for l in reversed(layer_list):
    print(l.name)
    decoder_output = l(decoder_output)
  de = Model(decoder_input, decoder_output)
  return de   


def make_decoder_input(shape):
  return Input(shape=(2,))


def decode_pic(decoder, x, y, num_w, Gs, Gs_kwargs):
  zij = decoder.predict(np.array([[x,y]]))
  zij = np.repeat(zij, num_w, axis=0).reshape(1, num_w, -1)
  num = Gs.components.synthesis.run(zij, **Gs_kwargs)[0]
  return num


def check_pic(full, i, j, points, pic_size=512):
  return full[pic_size*len(points)-(i*pic_size+pic_size):pic_size*len(points)-i*pic_size, j*pic_size:j*pic_size+pic_size]


def get_model_callbacks(model_name, patience=100):
  return EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience), ModelCheckpoint(f'models/best_{model_name}.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)

