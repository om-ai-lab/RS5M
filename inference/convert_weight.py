import torch
import open_clip
import os


def main():
    # trained_ckpt_path = "/home/zilun/RS5M_v5/ckpt/epoch_5.pt"
    # model, _, _ = open_clip.create_model_and_transforms("ViT-B/32", pretrained="openai")

    trained_ckpt_path = "/home/zilun/RS5M_v5/ckpt/epoch_2.pt"
    model, _, _ = open_clip.create_model_and_transforms("ViT-H/14", pretrained="openclip")

    checkpoint = torch.load(trained_ckpt_path, map_location="cpu")["state_dict"]
    sd = {k: v for k, v in checkpoint.items()}
    for key in list(sd.keys()):
        if "text_backbone." in key:
            sd[key.replace("text_backbone.", '')] = sd[key]
            del sd[key]
        if "image_backbone" in key:
            sd[key.replace("image_backbone.", "visual.")] = sd[key]
            del sd[key]

    msg = model.load_state_dict(sd, strict=False)
    print(msg)
    print("loaded RSCLIP")

    torch.save(
        model.state_dict(),
        os.path.join("/home/zilun/RS5M_v5/ckpt", "RS5M_ViT-B-32.pt"),
    )


if __name__ == "__main__":
    main()