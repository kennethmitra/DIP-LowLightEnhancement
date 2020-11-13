import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from LossFunctions import *
from ImageDataset import ImageDataset
from Model import EnhancerModel
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def train():
    writer = SummaryWriter()
    
    # ============================
    #       HYPERPARAMETERS
    # ============================
    #lr = 1e-4
    #weight_decay = 1e-4
    lr = 1e-3
    weight_decay = 0
    epochs = 200
    grad_clip = 0.1
    do_grad_clip = False
    batch_size = 8
    seed = 69
    FORCE_CPU = False

    run_name = "new_WB_loss_test2"
    save_dir = f'./saves/{run_name}'
    SAVE_EPOCH_FREQ = 1

    ILL_LOSS_WEIGHT = 7.5
    SPA_LOSS_WEIGHT = 8
    COL_LOSS_WEIGHT = 1.2
    EXP_LOSS_WEIGHT = 5.5
    COLVAR_LOSS_WEIGHT = 5

    # Create save directory if it doesn't exist
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Write tensorboard dir to file for future reference
    with open(f"{save_dir}/tensorboard.txt", "a+") as f:
        f.write(f"{writer.get_logdir()}\n")

    # Get compute device
    print("-------------------------------GPU INFO--------------------------------------------")
    print('Available devices ', torch.cuda.device_count())
    device = torch.device("cuda:0" if torch.cuda.is_available() and not FORCE_CPU else "cpu")
    print('Current cuda device ', device)
    if device != torch.device("cpu"):
        print('Current CUDA device name ', torch.cuda.get_device_name(device))
        torch.cuda.manual_seed(seed)
    print("-----------------------------------------------------------------------------------")

    # Set seed
    torch.manual_seed(seed)



    # Progress Pictures
    PROGRESS_PICS = None
    progpic_dataset = ImageDataset(image_dir="./images/progress_pics", image_dim=256, image_list=PROGRESS_PICS)
    progpic_dataloader = DataLoader(progpic_dataset, batch_size=batch_size)

    # Log parameters
    writer.add_text("Info/RUN_NAME", str(run_name), 0)
    writer.add_text("Info/progress_pics", str(PROGRESS_PICS), 0)
    writer.add_text("Hyperparams/lr", str(lr), 0)
    writer.add_text("Hyperparams/seed", str(seed), 0)
    writer.add_text("Hyperparams/do_grad_clip", str(do_grad_clip), 0)
    writer.add_text("Hyperparams/weight_decay", str(weight_decay), 0)
    writer.add_text("Hyperparams/batch_size", str(batch_size), 0)
    writer.add_text("Hyperparams/epochs", str(epochs), 0)
    writer.add_text("Hyperparams/grad_clip", str(grad_clip), 0)

    writer.add_text("Hyperparams/ill_loss_weight", str(ILL_LOSS_WEIGHT), 0)
    writer.add_text("Hyperparams/spa_loss_weight", str(SPA_LOSS_WEIGHT), 0)
    writer.add_text("Hyperparams/col_loss_weight", str(COL_LOSS_WEIGHT), 0)
    writer.add_text("Hyperparams/exp_loss_weight", str(EXP_LOSS_WEIGHT), 0)
    writer.add_text("Hyperparams/colvar_loss_weight", str(COLVAR_LOSS_WEIGHT), 0)

    writer.add_text("Info/FORCE_CPU", str(FORCE_CPU), 0)

    # Create Model
    model = EnhancerModel()
    model.to(device)
    
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Initialize weights from gaussian
    model.apply(weights_init)
    

    # Loss Functions
    color_loss = ColorConstancyLoss(method=3, device=device, patch_size=16)  # Using my custom WB loss
    exposure_loss = ExposureControlLoss(gray_value=0.5, patch_size=16, method=1, device=device)   # Using method 2 based on bsun0802's code
    spatial_loss = SpatialConsistencyLoss(device=device)
    illumination_loss = IlluminationSmoothnessLoss(method=3)  # From bsun0802's code
    colvar_loss = ColorVarianceLoss()
    
    # Datasets
    train_dataset = ImageDataset(image_dir="./images/train_data", image_dim=256)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)



    print(f"Number of training images: {len(train_dataset)}")

    for epoch in range(1, epochs + 1):

        loss_batch = []
        loss_batch_ill = []
        loss_batch_spa = []
        loss_batch_col = []
        loss_batch_exp = []
        loss_batch_colvar = []
        for batch_num, image in enumerate(train_dataloader):
            image = image.to(device)
            curves = model(image)
            enhanced_image = model.enhance_image(image, curves)
            
            illum_loss_val      = ILL_LOSS_WEIGHT * torch.mean(illumination_loss(curves))
            spatial_loss_val    = SPA_LOSS_WEIGHT * torch.mean(spatial_loss(enhanced_image, image))
            color_loss_val      = COL_LOSS_WEIGHT * torch.mean(color_loss(enhanced_image))
            exposure_loss_val   = EXP_LOSS_WEIGHT * torch.mean(exposure_loss(enhanced_image))
            colvar_loss_val     = COLVAR_LOSS_WEIGHT * torch.mean(colvar_loss(enhanced_image))

            loss = illum_loss_val + spatial_loss_val + color_loss_val + exposure_loss_val + colvar_loss_val

            optimizer.zero_grad()
            loss.backward()

            if do_grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            # Metrics
            loss_batch.append(loss.detach().item())
            loss_batch_ill.append(illum_loss_val.detach().item())
            loss_batch_spa.append(spatial_loss_val.detach().item())
            loss_batch_col.append(color_loss_val.detach().item())
            loss_batch_exp.append(exposure_loss_val.detach().item())
            loss_batch_colvar.append(colvar_loss_val.detach().item())


        epoch_loss_avg = np.asarray(loss_batch).mean()
        epoch_loss_ill_avg = np.asarray(loss_batch_ill).mean()
        epoch_loss_spa_avg = np.asarray(loss_batch_spa).mean()
        epoch_loss_col_avg = np.asarray(loss_batch_col).mean()
        epoch_loss_exp_avg = np.asarray(loss_batch_exp).mean()
        epoch_loss_colvar_avg = np.asarray(loss_batch_colvar).mean()

        # Checkpoint
        if epoch % SAVE_EPOCH_FREQ == 0 or epoch == epochs:

            # Save model/optimizer state
            torch.save({
                'epoch': epoch,
                'loss': epoch_loss_avg,
                'optimizer_params': optimizer.state_dict(),
                'model_state': model.state_dict(),
                }, f'{save_dir}/epo{epoch}.save')
            # Generate Progress Pics
            # progress_images = None
            # with torch.no_grad():
            #     for batch in progpic_dataloader:
            #         batch = batch.to(device)
            #         curves = model(batch)
            #         enhanced_images = model.enhance_image(batch, curves)
            #
            #         if progress_images is None:
            #             progress_images = enhanced_images
            #         else:
            #             progress_images = torch.cat([progress_images, enhanced_images], dim=0)
            #
            # # Convert images to grid
            # grid_img = torchvision.utils.make_grid(progress_images, padding=2, normalize=False)

            # Save images
            # torchvision.utils.save_image(grid_img, f'{save_dir}/progress_pics_epo{epoch}.png')
            # writer.add_image("fixed_inputs", grid_img, epoch)
            progpic_filepath = f"{save_dir}/progress_epo{epoch}.png"
            plt_tensor = create_progress_pics(model, device, progpic_dataset, save_file_path=progpic_filepath)
            writer.add_image("Progress_Pics", plt_tensor, epoch)

        print(f"Finished Epoch {epoch} / {epochs} \t | \t Loss: {epoch_loss_avg}")

        writer.add_scalar("Metrics/Overall_Loss", epoch_loss_avg, epoch)
        writer.add_scalar("Metrics/Illumination_Loss", epoch_loss_ill_avg, epoch)
        writer.add_scalar("Metrics/Spatial_Loss", epoch_loss_spa_avg, epoch)
        writer.add_scalar("Metrics/Color_Loss", epoch_loss_col_avg, epoch)
        writer.add_scalar("Metrics/Exposure_Loss", epoch_loss_exp_avg, epoch)
        writer.add_scalar("Metrics/ColorVariance_Loss", epoch_loss_colvar_avg, epoch)

def create_progress_pics(model, device, progress_ds, save_file_path):
    fig, ax = plt.subplots(len(progress_ds), 4, figsize=(17, 17))

    with torch.no_grad():
        for img_num, image in enumerate(progress_ds):
            image = image.unsqueeze(dim=0).to(device)  # Add batch dimension and send to GPU
            curves = model(image)
            enhanced_image = model.enhance_image(image, curves)

            curves = torch.stack(torch.split(curves, split_size_or_sections=3, dim=1), dim=1)

            image = image.squeeze().permute(1, 2, 0).cpu()
            curves = (curves.squeeze().permute(0, 2, 3, 1).mean(dim=0).cpu())/ 2 + 0.5
            enhanced_image = enhanced_image.squeeze().permute(1, 2, 0).cpu()

            ax[img_num][0].imshow(image)
            ax[img_num][0].set_title("Original")
            ax[img_num][1].imshow(curves)
            ax[img_num][1].set_title("RGB Curves")
            ax[img_num][2].imshow(curves.mean(dim=2))
            ax[img_num][2].set_title("Grayscale Curves")
            ax[img_num][3].imshow(enhanced_image)
            ax[img_num][3].set_title("Enhanced Image")

    fig.savefig(save_file_path)
    plt.close(fig)
    return torch.from_numpy(np.array(Image.open(save_file_path))[:, :, :3]).permute(2, 0, 1)


if __name__ == "__main__":
    train()
