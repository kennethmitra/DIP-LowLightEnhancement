import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from LossFunctions import *
from ImageDataset import ImageDataset
from Model import EnhancerModel
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import torchvision


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def train():
    writer = SummaryWriter()

    # Get compute device
    print("-------------------------------GPU INFO--------------------------------------------")
    print('Available devices ', torch.cuda.device_count())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Current cuda device ', device)
    if device != "cpu":
        print('Current CUDA device name ', torch.cuda.get_device_name(device))
    print("-----------------------------------------------------------------------------------")
    
    # ============================
    #       HYPERPARAMETERS
    # ============================
    lr = 1e-4
    weight_decay = 1e-4
    epochs = 200
    grad_clip = 0.1
    batch_size = 8

    run_name = "orig_code"
    save_dir = f'./saves/{run_name}'
    SAVE_EPOCH_FREQ = 1

    ILL_LOSS_WEIGHT = 200
    SPA_LOSS_WEIGHT = 1
    COL_LOSS_WEIGHT = 5
    EXP_LOSS_WEIGHT = 10

    # Progress Pictures
    PROGRESS_PICS = ['2.jpg', '7.jpg', '11.jpg', '639.jpg']
    progpic_dataset = ImageDataset(image_dir="./images/train_data", image_dim=512, image_list=PROGRESS_PICS)
    progpic_dataloader = DataLoader(progpic_dataset, batch_size=batch_size)

    # Log parameters
    writer.add_text("Info/RUN_NAME", str(run_name), 0)
    writer.add_text("Info/progress_pics", str(PROGRESS_PICS), 0)
    writer.add_text("Hyperparams/lr", str(lr), 0)
    writer.add_text("Hyperparams/weight_decay", str(weight_decay), 0)
    writer.add_text("Hyperparams/batch_size", str(batch_size), 0)
    writer.add_text("Hyperparams/epochs", str(epochs), 0)
    writer.add_text("Hyperparams/grad_clip", str(grad_clip), 0)

    writer.add_text("Hyperparams/ill_loss_weight", str(ILL_LOSS_WEIGHT), 0)
    writer.add_text("Hyperparams/spa_loss_weight", str(SPA_LOSS_WEIGHT), 0)
    writer.add_text("Hyperparams/col_loss_weight", str(COL_LOSS_WEIGHT), 0)
    writer.add_text("Hyperparams/exp_loss_weight", str(EXP_LOSS_WEIGHT), 0)

    # Create Model
    model = EnhancerModel()
    model.to(device)
    
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Initialize weights from gaussian
    model.apply(weights_init)
    

    # Loss Functions
    color_loss = ColorConstancyLoss(method=1)
    exposure_loss = ExposureControlLoss(gray_value=0.6, patch_size=16, method=1)
    spatial_loss = SpatialConsistencyLoss(device=device)
    illumination_loss = IlluminationSmoothnessLoss(method=1)
    
    # Datasets
    train_dataset = ImageDataset(image_dir="./images/train_data", image_dim=256)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Create save directory if it doesn't exist
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    print(f"Number of training images: {len(train_dataset)}")

    for epoch in range(1, epochs + 1):

        loss_batch = []
        loss_batch_ill = []
        loss_batch_spa = []
        loss_batch_col = []
        loss_batch_exp = []
        for batch_num, image in enumerate(train_dataloader):
            image = image.to(device)
            curves = model(image)
            enhanced_image = model.enhance_image(image, curves)
            
            illum_loss_val      = ILL_LOSS_WEIGHT * torch.mean(illumination_loss(image))
            spatial_loss_val    = SPA_LOSS_WEIGHT * torch.mean(spatial_loss(enhanced_image, image))
            color_loss_val      = COL_LOSS_WEIGHT * torch.mean(color_loss(enhanced_image))
            exposure_loss_val   = EXP_LOSS_WEIGHT * torch.mean(exposure_loss(enhanced_image))

            loss = illum_loss_val + spatial_loss_val + color_loss_val + exposure_loss_val

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            # Metrics
            loss_batch.append(loss.detach().item())
            loss_batch_ill.append(illum_loss_val.detach().item())
            loss_batch_spa.append(spatial_loss_val.detach().item())
            loss_batch_col.append(color_loss_val.detach().item())
            loss_batch_exp.append(exposure_loss_val.detach().item())


        epoch_loss_avg = np.asarray(loss_batch).mean()
        epoch_loss_ill_avg = np.asarray(loss_batch_ill).mean()
        epoch_loss_spa_avg = np.asarray(loss_batch_spa).mean()
        epoch_loss_col_avg = np.asarray(loss_batch_spa).mean()
        epoch_loss_exp_avg = np.asarray(loss_batch_exp).mean()

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
            progress_images = None
            with torch.no_grad():
                for batch in progpic_dataloader:
                    batch = batch.to(device)
                    curves = model(batch)
                    enhanced_images = model.enhance_image(batch, curves)

                    if progress_images is None:
                        progress_images = enhanced_images
                    else:
                        progress_images = torch.cat([progress_images, enhanced_images], dim=0)

            # Convert images to grid
            grid_img = torchvision.utils.make_grid(progress_images, padding=2, normalize=False)

            # Save images
            torchvision.utils.save_image(grid_img, f'{save_dir}/progress_pics_epo{epoch}.png')
            writer.add_image("fixed_inputs", grid_img, epoch)




        print(f"Finished Epoch {epoch} / {epochs} \t | \t Loss: {epoch_loss_avg}")

        writer.add_scalar("Metrics/Overall_Loss", epoch_loss_avg, epoch)
        writer.add_scalar("Metrics/Illumination_Loss", epoch_loss_ill_avg, epoch)
        writer.add_scalar("Metrics/Spatial_Loss", epoch_loss_spa_avg, epoch)
        writer.add_scalar("Metrics/Color_Loss", epoch_loss_col_avg, epoch)
        writer.add_scalar("Metrics/Exposure_Loss", epoch_loss_exp_avg, epoch)

if __name__ == "__main__":
    train()
