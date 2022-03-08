from keras.models import Model
import pretrained_networks
import dnnlib
import dnnlib.tflib as tflib


def load_network(network_pkl):
  _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
  noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

  Gs_kwargs = dnnlib.EasyDict()
  Gs_kwargs.randomize_noise = False
  Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)

  return _G, _D, Gs, Gs_kwargs


def convertZtoW(Gs, latent, truncation_psi=0.7, truncation_cutoff=9):
    dlatent = Gs.components.mapping.run(latent, None) # [seed, layer, component]
    dlatent_avg = Gs.get_var('dlatent_avg') # [component]
    dlatent = dlatent_avg + (dlatent - dlatent_avg) * truncation_psi

    return dlatent