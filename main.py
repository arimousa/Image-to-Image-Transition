from PIL import Image
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from diffusers import UniPCMultistepScheduler
from tqdm.auto import tqdm
import numpy as np
from torchvision import transforms
import cv2
import glob


def compute_alpha(t):
    trajectory_steps = 1000
    beta_start = 0.0001
    beta_end = 0.02 
    betas = np.linspace(beta_start, beta_end, trajectory_steps, dtype=np.float64)
    betas = torch.tensor(betas).type(torch.float)
    beta = torch.cat([torch.zeros(1).to(betas.device), betas], dim=0)
    beta = beta.to('cuda')
    t = t.to('cuda')
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

if __name__ == "__main__":

    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="text_encoder")
    unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")


    scheduler = UniPCMultistepScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

    torch_device = "cuda"
    vae.to(torch_device)
    text_encoder.to(torch_device)
    unet.to(torch_device)


    prompt = ["Impressionist oil painting of a cute robot"]
    height = 512  # default height of Stable Diffusion
    width = 512  # default width of Stable Diffusion
    num_inference_steps = 100  # Number of denoising steps
    guidance_scale = 7.5  # Scale for classifier-free guidance
    generator = torch.manual_seed(5)  # Seed generator to create the inital latent noise
    batch_size = len(prompt)


    text_input = tokenizer(
        prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
    )

    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]


    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])


    latents = torch.randn(
        (batch_size, unet.in_channels, height // 8, width // 8),
        generator=generator,
    )
    latents = latents.to(torch_device)
    latents = latents * scheduler.init_noise_sigma


    #reading the target iamge
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )


    target_image = Image.open("target.png")
    target_image = preprocess(target_image)
    target_image = target_image.unsqueeze(0).to(torch_device)
    latent_target = vae.encode(target_image).latent_dist.sample() * 0.18215


    # input_image = Image.open("input.png")
    # input_image = preprocess(input_image)
    # input_image = input_image.unsqueeze(0).to(torch_device)
    # latent_input0 = vae.encode(input_image).latent_dist.sample() * 0.18215
    # pre_latents = latent_input0

    latents0 = latents
    w = 0
    w2 = 0.05
    increase = True




    scheduler.set_timesteps(num_inference_steps)

    for j in range(1000):
        latents = latents0

        for t in scheduler.timesteps:
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            
        

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            at = compute_alpha(t.long())
            # if t > 100:
            #     latent_target_t = scheduler.add_noise(latent_target,noise_pred_uncond ,t-80)
            # else:
            latent_target_t = scheduler.add_noise(latent_target,noise_pred_uncond ,t//2)
            if j == 0:
                noise_pred = noise_pred_uncond  - (1 - at).sqrt() * (w) * (latent_target_t-latents)  + guidance_scale * (noise_pred_text - noise_pred_uncond) 
            else:
                latent_input_t =  scheduler.add_noise(input_latent,noise_pred_uncond ,t//2)
                noise_pred = noise_pred_uncond  - (1 - at).sqrt() * (w) * (latent_target_t-latents)  + guidance_scale * (noise_pred_text - noise_pred_uncond) - (1 - at).sqrt() * (w2) * (latent_input_t-latents) 
            
            # latent_input_t = at.sqrt() * pre_latents + (1- at).sqrt() * noise_pred

            # noise_pred = noise_pred  - (1 - at).sqrt() * w2 * (latent_input_t - latents) + guidance_scale * (noise_pred_text - noise_pred_uncond) 
            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents).prev_sample


            # noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)  - (1 - at).sqrt() * w * (latent_target_t-latents)

            # # compute the previous noisy sample x_t -> x_t-1
            # latents = scheduler.step(noise_pred, t, latents).prev_sample


        # update the latents for the next iteration
        if j == 0:
            print("first")
            input_latent = latents
            # input_latent = latents0
        # else:
        #     latents0 = latents0 + (latents - pre_latents) * 0.005
        #     # latents0 = latents0 - diff
        #     pre_latents = latents
        #     latents = latents0



        # scale and decode the image latents with vae
        latents_de = 1 / 0.18215 * latents
        with torch.no_grad():
            image = vae.decode(latents_de).sample

        
            



        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        pil_images[0].save("vid/{:.4f}.png".format(w))
        
        
        if w > 0.35:
            increase = False
        if increase == True:
            w += 0.0015
            w2 += 0.0005
        elif increase == False and w2>0:
            w += 0.001
            w2 -= 0.0005
        


        # w2 -= 0.005
        # guidance_scale = guidance_scale - 0.0001	





    
    # Initialize the video writer
    output_video = cv2.VideoWriter("output_video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (512, 512))
    image_paths = sorted(glob.glob("vid/*.png"))
    # Process each frame and write it to the video
    for image_path in image_paths:
        # Load and preprocess the image
        image = cv2.imread(image_path)
        image_tensor = preprocess(image).unsqueeze(0).to('cuda')

        # Process the frame (perform any desired operations)
        # ...

        # Convert the frame tensor back to an image
        frame_image = (image_tensor.squeeze().cpu() * 0.5 + 0.5).clamp(0.0, 1.0).numpy().transpose((1, 2, 0)) * 255
        frame_image = frame_image.astype("uint8")

        # Write the frame to the video
        output_video.write(frame_image)

    # Release the video writer
    output_video.release()



















    # scheduler.set_timesteps(num_inference_steps)


    # t_input = 0
    # for j in range(num_inference_steps):
    #     for t in scheduler.timesteps:
    #         # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    #         if t == t_input:
    #             latents = scheduler.add_noise(latent_input0, latents0 ,t)
    #         latent_model_input = torch.cat([latents] * 2)

    #         latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

    #         # predict the noise residual
    #         with torch.no_grad():
    #             noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            
        

    #         # perform guidance
    #         noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)


    #         test_trajectoy_steps = torch.Tensor([t]).type(torch.int64).to('cuda')
    #         at = compute_alpha(test_trajectoy_steps.long())
    #         latent_target_t = scheduler.add_noise(latent_target, noise_pred_uncond ,t)  #at.sqrt() * latent_target + (1- at).sqrt() *  noise_pred_uncond 
            

 
    #         noise_pred = noise_pred_uncond  - (1 - at).sqrt() * w * (latent_target_t - latents)  #+ guidance_scale * (noise_pred_text - noise_pred_uncond) 

    #         latent_input_t = scheduler.add_noise(pre_latents, noise_pred ,t)  #at.sqrt() * pre_latents + (1- at).sqrt() * noise_pred

    #         noise_pred = noise_pred  - (1 - at).sqrt() * w2 * (latent_input_t - latents) #+ guidance_scale * (noise_pred_text - noise_pred_uncond) 
    #         # compute the previous noisy sample x_t -> x_t-1
    #         latents = scheduler.step(noise_pred, t, latents).prev_sample



    #     # scale and decode the image latents with vae
    #     latents_de = 1 / 0.18215 * latents
    #     with torch.no_grad():
    #         image = vae.decode(latents_de).sample

    #     # update the latents for the next iteration
    #     if j == 0:
    #         pre_latents = latents
    #         latents = latents0
    #     else:
    #         latents0 = latents0 + (latents - pre_latents) * 0.005
    #         # latents0 = latents0 - diff
    #         pre_latents = latents
    #         latents = latents0
            



    #     image = (image / 2 + 0.5).clamp(0, 1)
    #     image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    #     images = (image * 255).round().astype("uint8")
    #     pil_images = [Image.fromarray(image) for image in images]
    #     pil_images[0].save("vid/{:.4f}.png".format(w))
    #     w+=0.005
    #     w2 -= 0.005
    #     t_input += 1000 // num_inference_steps


    #     # guidance_scale = guidance_scale - 0.0001	


########################################################################################################################################
