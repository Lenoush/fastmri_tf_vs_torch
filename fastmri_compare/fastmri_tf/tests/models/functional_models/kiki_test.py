from fastmri_tf.models.functional_models.kiki import kiki_net
from fastmri_tf.models.utils.non_linearities import lrelu


def test_init_kiki():
    run_params = {
        'n_cascade': 2,
        'n_convs': 25,
        'n_filters': 32,
        'noiseless': True,
        'activation': lrelu,
    }
    model = kiki_net(lr=1e-3, **run_params)
