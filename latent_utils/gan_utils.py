from keras.models import Model
import dnnlib
import dnnlib.tflib as tflib


def convertZtoW(Gs, latent, truncation_psi=0.7, truncation_cutoff=9):
    dlatent = Gs.components.mapping.run(latent, None) # [seed, layer, component]
    dlatent_avg = Gs.get_var('dlatent_avg') # [component]
    dlatent = dlatent_avg + (dlatent - dlatent_avg) * truncation_psi

    return dlatent