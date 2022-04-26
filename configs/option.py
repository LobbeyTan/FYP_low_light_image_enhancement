import os
from re import S


class Option:

    def __init__(self, root_dir=os.getcwd(), phase="train") -> None:
        self.phase = phase
        self.root_dir = root_dir


class BaseOptions():

    def __init__(
        self,
        dataroot,
        batch_size,
        load_size,
        fine_size,
        patch_size,
        input_nc,
        output_nc,
        ngf,
        ndf,
        netD_type,
        netG_type,
        n_layers_D,
        n_layers_patchD,
        device,
        name,
        dataset_direction,
        checkpoints_dir,
        norm_type,
        identity,
        no_dropout,
        lambda_A,
        lambda_B,
        preprocessing,
        no_flip,
        skip,
        use_mse,
        l1_weight,
        use_norm,
        use_wgan,
        use_ragan,
        use_vgg,
        apply_vgg_mean,
        vgg_choose,
        no_vgg_instance,
        vgg_maxpooling,
        in_vgg,
        fcn,
        use_avgpool,
        use_instance_norm,
        use_syn_norm,
        use_tanh,
        use_linear,
        apply_noise,
        latent_threshold,
        latent_norm,
        use_patchD,
        n_patch,
        double_patch_D_loss,
        use_patch_vgg,
        hybrid_loss,
        self_attention,
        times_residual,
        norm_attention,
        apply_vary,
    ):
        """
            Args:
                @param dataroot (str): path to images (should have subfolders trainA, trainB, valA, valB, etc) `required=True`

                @param batch_size (int): input batch size `default=1`

                @param load_size (int): scale images to this size `default=286`

                @param fine_size (int): crop images to this size `default=256`

                @param patch_size (int): crop patch images to this size `default=64`

                @param input_nc (int): number of input image channels `default=3`

                @param output_nc (int): number of output image channels `default=3`

                @param ngf (int): number of generator filters in first conv layer `default=64`

                @param ndf (int): number of discriminator filters in first conv layer `default=64`

                @param netD_type (str): selects model to use for netD `default=NLayerDiscriminator`

                @param netG_type (str): selects model to use for netG `default=Resnet`

        """
        self.dataroot
        self.batch_size
        self.load_size
        self.fine_size
        self.patch_size
        self.input_nc
        self.output_nc
        self.ngf
        self.ndf
        self.netD_type
        self.netG_type
        self.n_layers_D
        self.n_layers_patchD
        self.device
        self.name
        self.dataset_direction
        self.checkpoints_dir
        self.norm_type
        self.identity
        self.no_dropout
        self.lambda_A
        self.lambda_B
        self.preprocessing
        self.no_flip
        self.skip
        self.use_mse
        self.l1_weight
        self.use_norm
        self.use_wgan
        self.use_ragan
        self.use_vgg
        self.apply_vgg_mean
        self.vgg_choose
        self.no_vgg_instance
        self.vgg_maxpooling
        self.in_vgg,
        self.fcn
        self.use_avgpool
        self.use_instance_norm
        self.use_syn_norm
        self.use_tanh
        self.use_linear
        self.apply_noise
        self.latent_threshold
        self.latent_norm
        self.use_patchD
        self.n_patch
        self.double_patch_D_loss
        self.use_patch_vgg
        self.hybrid_loss
        self.self_attention
        self.times_residual
        self.norm_attention
        self.apply_vary
