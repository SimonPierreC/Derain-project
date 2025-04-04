import tools_preprocess_image
import tools_gmm
import tools_opti


def remove_rain_streaks(image_path, new_x=None, new_y=None,
                        alpha=1e-8, beta=1e-9, gamma=1e-9, omega0=1,
                        epsilon=1e-10, max_iter=20):
    Y, Cr, Cb = tools_preprocess_image.preprocess_image(
        image_path, new_x, new_y)
    dim_b, nmodels_b, means_b, covs_b, invcovs_b, mixweights_b = tools_gmm.load_background_gmm(
        'data/GSModel_8x8_200_2M_noDC_zeromean.mat')
    b_gmm = tools_gmm.init_gmm(
        nmodels_b, means_b, covs_b, invcovs_b, mixweights_b)

    (r_gmm, dim_r, nmodels_r, means_r, covs_r, invcovs_r,
     mixweights_r), region = tools_gmm.fit_rain_gmm(Y, region_size=(50, 50))
    B, R, _, _ = tools_opti.opti(Y,
                                 r_gmm=r_gmm, b_gmm=b_gmm,
                                 alpha=alpha, beta=beta, gamma=gamma,
                                 omega0=omega0,
                                 epsilon=epsilon, max_iter=max_iter)

    return tools_preprocess_image.to_RGB(B*255, Cr, Cb), R
