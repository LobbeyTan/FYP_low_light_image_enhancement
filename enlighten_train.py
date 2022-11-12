from data.custom_image_dataset import CustomImageDataset
from models.enlighten import EnlightenGAN
from configs.option import Option
from torch.utils.data import DataLoader

import logging
import torch
import time

if __name__ == "__main__":
    nth_exp = 41

    logging.basicConfig(
        filemode="w",
        level=logging.INFO,
        filename=f"./logs/EnlightenGAN/training_{nth_exp}.log",
        format="%(asctime)s - [%(levelname)s] - %(message)s",
        datefmt='%H:%M:%S',
    )

    logging.info("====================================================")
    logging.info((f"Training {nth_exp}", "Grayscale low light image enhancement"))
    logging.info("====================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device used:", device)
    logging.info(f"Device used: {device}")

    # Configurations

    img_dir = "./datasets/light_enhancement"
    checkpoint_dir = f"./checkpoints/enlightenGAN/training_{nth_exp}/"
    batch_size = 16
    batch_shuffle = True

    lr = 0.0001

    n_epochs = 300
    print_freq = 1000
    save_freq = 15000

    # Load dataset & dataloader

    dataset = CustomImageDataset(
        img_dir=img_dir,
        opt=Option(phase="train", grayscale=True)
    )

    dataloader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=batch_shuffle,
    )

    dataloader_size = len(dataloader)

    print("The number of training images = %d" % dataloader_size)
    logging.info(f"The number of training images = {dataloader_size}")

    # Load model
    model = EnlightenGAN(
        use_src=False, lr=lr, device=device
    )

    # Start training
    total_iterations = 0
    train_start_time = time.time()

    n_print = 1
    n_save = 1

    print("====================Start Training:====================")
    logging.info("====================Start Training:====================")

    for epoch in range(n_epochs):
        start_time = time.time()

        epoch_iter = 0

        for i, data in enumerate(dataloader):
            model.set_input(data)
            model.optimize_parameters()

            total_iterations += len(data['img_A'])
            epoch_iter += len(data['img_A'])

            if total_iterations > (print_freq * n_print):
                time_taken = time.time() - train_start_time

                logging.info(
                    "--------------------E%d-----------------------" % (epoch+1))
                logging.info("Current Iteration: %05d | Epoch Iteration: %05d" %
                             (print_freq * n_print, epoch_iter))
                logging.info("Current Time Taken: %07ds | Current Epoch Running Time: %07ds" % (
                    time_taken, time.time() - start_time))
                logging.info("SPA Loss: %.7f | Color Loss: %.7f" %
                             (model.loss_spa, model.loss_color))
                logging.info("RAGAN Loss for Global D: %.7f | Local D: %.7f" %
                             (model.loss_D, model.loss_patch_D))
                logging.info("RAGAN Loss for Global G: %.7f | Local G: %.7f" %
                             (model.loss_G, model.loss_G_patch))
                logging.info("SFP Loss for Global G  : %.7f | Local G: %.7f" %
                             (model.loss_G_SFP, model.loss_G_SFP_patch))
                logging.info(f"Total generator loss: {model.total_loss_G}")
                n_print += 1

            if total_iterations > (save_freq * n_save):
                logging.info("Saving models...")
                model.save_model(checkpoint_dir, save_freq * n_save)
                n_save += 1

    print(f"Total time taken: {time.time() - train_start_time}")
    print("Saving trained model ...")

    logging.info(f"Total time taken: {time.time() - train_start_time}")
    logging.info("Saving trained model ...")

    model.save_model(checkpoint_dir, epoch="trained")

    print("====================End Training:====================")
    logging.info("====================End Training:====================")
