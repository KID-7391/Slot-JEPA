import os
import sys
import torch
import torch.nn as nn
from functools import partial


# Dummy functions and classes to handle pickle issues when loading certain checkpoints
# These are placeholders for functions/classes that were defined in the original training scripts

def exclude_bias_and_norm(p):
    """Dummy function for VICReg/VICRegL/BarlowTwins checkpoints."""
    return p.ndim == 1

def adjust_learning_rate(args, optimizer, loader, step):
    """Dummy function for VICReg/VICRegL/BarlowTwins checkpoints."""
    pass

def batch_all_gather(x):
    """Dummy function for distributed training checkpoints."""
    return x

class LARS(torch.optim.Optimizer):
    """
    Dummy LARS optimizer class for checkpoint unpickling.
    This is a placeholder - the actual optimizer state is not needed for inference.
    """
    def __init__(self, params, lr=0, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=None, lars_adaptation_filter=None):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    def step(self, closure=None):
        pass


def _register_pickle_helpers():
    """Register dummy functions/classes in __main__ to help with checkpoint unpickling."""
    import __main__
    if not hasattr(__main__, 'exclude_bias_and_norm'):
        __main__.exclude_bias_and_norm = exclude_bias_and_norm
    if not hasattr(__main__, 'adjust_learning_rate'):
        __main__.adjust_learning_rate = adjust_learning_rate
    if not hasattr(__main__, 'batch_all_gather'):
        __main__.batch_all_gather = batch_all_gather
    if not hasattr(__main__, 'LARS'):
        __main__.LARS = LARS


def _is_slim_checkpoint(checkpoint, model_type=None):
    if not isinstance(checkpoint, dict):
        return False
    if "state_dict" not in checkpoint or "model_type" not in checkpoint:
        return False
    if model_type is None:
        return True
    return checkpoint.get("model_type") == model_type


def extract_encoder_weights(checkpoint_path, model_type, output_path=None):
    """
    Extract only the encoder state_dict from a full checkpoint and save it.
    This saves space by removing optimizer states, teacher weights, etc.
    
    Args:
        checkpoint_path: Path to the original checkpoint
        model_type: Type of SSL model (e.g., 'mocov3', 'dino', 'byol', etc.)
        output_path: Path to save the slim checkpoint. If None, saves to same location with '_slim' suffix.
    
    Returns:
        Path to the saved slim checkpoint
    """
    if output_path is None:
        base, ext = os.path.splitext(checkpoint_path)
        output_path = f"{base}_slim{ext}"
    
    # Register dummy functions to handle pickle issues with certain checkpoints
    _register_pickle_helpers()
    
    # weights_only=False: raw checkpoints may contain pickled optimizer/scheduler objects
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if model_type == 'mae':
        state_dict = checkpoint['model']
    
    elif model_type == 'dino':
        state_dict = checkpoint['teacher']
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items() if k.startswith('backbone.')}
    
    elif model_type == 'ibot' or model_type == 'mugs':
        state_dict = checkpoint['teacher']
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items() if k.startswith('backbone.')}
    
    elif model_type == 'ijepa':
        state_dict = checkpoint['encoder']
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    elif model_type == 'mocov3' or model_type == 'RINCE':
        state_dict = checkpoint['state_dict']
        state_dict = {k: v for k, v in state_dict.items() if k.startswith('module.momentum_encoder.')}
        state_dict = {k.replace("module.momentum_encoder.", ""): v for k, v in state_dict.items()}
    
    elif model_type == 'mec' or model_type == 'resa' or model_type == 'simsiam':
        state_dict = checkpoint['state_dict']
        state_dict = {k: v for k, v in state_dict.items() if k.startswith('module.encoder.')}
        state_dict = {k.replace("module.encoder.", ""): v for k, v in state_dict.items()}
    
    elif model_type == 'densecl':
        state_dict = checkpoint['state_dict']
        state_dict = {k: v for k, v in state_dict.items() if k.startswith('backbone.')}
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    
    elif model_type == 'swav':
        state_dict = checkpoint.get('state_dict', checkpoint)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    elif model_type == 'vicreg' or model_type == 'vicregl':
        state_dict = checkpoint['model']
        state_dict = {k: v for k, v in state_dict.items() if 'encoder' not in k}
        state_dict = {k.replace("module.backbone.", ""): v for k, v in state_dict.items()}
    
    elif model_type == 'byol':
        state_dict = checkpoint['model']
        state_dict = {k: v for k, v in state_dict.items() if k.startswith('module.target_network.')}
        state_dict = {k.replace("module.target_network.", ""): v for k, v in state_dict.items()}
    
    elif model_type == 'barlowtwins':
        state_dict = checkpoint['model']
        state_dict = {k: v for k, v in state_dict.items() if 'encoder' not in k}
        state_dict = {k.replace("module.backbone.", ""): v for k, v in state_dict.items()}
    
    elif model_type == 'esvit':
        state_dict = checkpoint['teacher']

    elif model_type == 'dinov2':
        # DINOv2 pretrained weights are already a bare encoder state_dict
        state_dict = checkpoint

    else:
        raise NotImplementedError(f"Model type '{model_type}' not supported for weight extraction")
    
    slim_checkpoint = {'state_dict': state_dict, 'model_type': model_type}
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(slim_checkpoint, output_path)
    
    return output_path


def get_models_slim(model_type, model_scale, model_root, epoch, device):
    """
    Load model from a slim checkpoint (processed by extract_encoder_weights).
    
    Slim checkpoints are stored in: {model_root}/{model_type}/{original_filename}
    
    Args:
        model_type: Type of SSL model
        model_scale: Model architecture scale
        model_root: Root directory containing slim checkpoints (e.g., /path/to/checkpoints_slim)
        epoch: Epoch number to load
        device: Device to load model on
    
    Returns:
        Loaded model
    """
    # Build path: {model_root}/{model_type}/{checkpoint_filename}
    type_dir = os.path.join(model_root, model_type)
    
    if model_type == 'mae':
        from .mae.models_mae import mae_vit_small_patch16 as vit_small
        model = vit_small().to(device)
        checkpoint_path = os.path.join(type_dir, f'checkpoint-{epoch}.pth')
    
    elif model_type == 'dino':
        from .dino.vision_transformer import vit_small
        if model_scale == 'vit_small':
            model = vit_small().to(device)
            checkpoint_path = os.path.join(type_dir, f'checkpoint{str(epoch).zfill(4)}.pth')
        else:
            raise NotImplementedError
    
    elif model_type == 'ibot' or model_type == 'mugs':
        from .ibot.vision_transformer import vit_small
        if model_scale == 'vit_small':
            model = vit_small().to(device)
            checkpoint_path = os.path.join(type_dir, f'checkpoint{str(epoch).zfill(4)}.pth')
        else:
            raise NotImplementedError
    
    elif model_type == 'ijepa':
        from .ijepa.deit import deit_base, deit_large
        if model_scale == 'vit_base':
            model = deit_base().to(device)
            checkpoint_path = os.path.join(type_dir, f'ijepa-ep{epoch}.pth.tar')
        else:
            raise NotImplementedError
    
    elif model_type == 'mocov3' or model_type == 'RINCE':
        from .mocov3.vit_moco import vit_small
        if model_scale == 'vit_small':
            model = vit_small().to(device)
            checkpoint_path = os.path.join(type_dir, f'checkpoint_{str(epoch).zfill(4)}.pth.tar')
        else:
            raise NotImplementedError
    
    elif model_type == 'mec':
        if model_scale == 'resnet50':
            from .resnet import resnet50
            model = resnet50().to(device)
            checkpoint_path = os.path.join(type_dir, f'checkpoint_{str(epoch).zfill(4)}.pth.tar')
        else:
            raise NotImplementedError

    elif model_type == 'resa':
        if model_scale == 'resnet50':
            from .resnet import resnet50
            model = resnet50().to(device)
            checkpoint_path = os.path.join(type_dir, f'checkpoint_{str(epoch).zfill(4)}.pth.tar')
        else:
            raise NotImplementedError

    elif model_type == 'simsiam':
        if model_scale == 'resnet50':
            from .resnet import resnet50
            model = resnet50().to(device)
            checkpoint_path = os.path.join(type_dir, f'checkpoint_{str(epoch).zfill(4)}.pth.tar')
        else:
            raise NotImplementedError

    elif model_type == 'densecl':
        if model_scale == 'resnet50':
            from .resnet import resnet50
            model = resnet50().to(device)
            checkpoint_path = os.path.join(type_dir, f'epoch_{epoch}.pth')
        else:
            raise NotImplementedError

    elif model_type == 'swav':
        if model_scale == 'resnet50':
            from .resnet import resnet50
            model = resnet50().to(device)
            checkpoint_path = os.path.join(type_dir, f'ckp-{epoch}.pth')
        else:
            raise NotImplementedError

    elif model_type == 'vicreg' or model_type == 'vicregl':
        if model_scale == 'resnet50':
            from .resnet import resnet50
            model = resnet50().to(device)
            checkpoint_path = os.path.join(type_dir, f'model_{epoch}.pth')
        else:
            raise NotImplementedError
    
    elif model_type == 'byol':
        from .byol.byol import EncoderwithProjection
        model = EncoderwithProjection().to(device)
        checkpoint_path = os.path.join(type_dir, f'resnet50_{str(epoch)}.pth.tar')
    
    elif model_type == 'barlowtwins':
        if model_scale == 'resnet50':
            from .resnet import resnet50
            model = resnet50().to(device)
            checkpoint_path = os.path.join(type_dir, f'checkpoint_{epoch}.pth')
        else:
            raise NotImplementedError

    elif model_type == 'esvit':
        from .esvit.swin_transformer import SwinTransformer
        if model_scale == 'swint_tiny':
            model = SwinTransformer(
                img_size=[224, 224],
                in_chans=3,
                num_classes=1000,
                patch_size=4,
                embed_dim=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                mlp_ratio=4,
                qkv_bias=True,
                drop_rate=0,
                attn_drop_rate=0,
                drop_path_rate=0.1,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                ape=False,
                patch_norm=True,
                use_dense_prediction=False,
            ).to(device)
            checkpoint_path = os.path.join(type_dir, f'checkpoint{str(epoch).zfill(4)}.pth')
        else:
            raise NotImplementedError

    elif model_type == 'dinov2':
        from .dinov2.vision_transformer import vit_small as dinov2_vit_small
        if model_scale == 'vit_small':
            model = dinov2_vit_small(patch_size=14, num_register_tokens=4).to(device)
            checkpoint_path = os.path.join(type_dir, 'dinov2_vits14_reg4_pretrain.pth')
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError(f"Model type '{model_type}' not supported")

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    state_dict = checkpoint['state_dict']
    # DINOv2: pos_embed may have different size (pretrained at 518x518, model at 224x224)
    if model_type == 'dinov2' and 'pos_embed' in state_dict:
        model.pos_embed = nn.Parameter(state_dict.pop('pos_embed'))
    msg = model.load_state_dict(state_dict, strict=False)

    print(msg)
    return model

def get_models(model_type, model_scale, model_root, epoch, device):
    if model_type == 'mae':
        from .mae.models_mae import mae_vit_small_patch16 as vit_small
        model = vit_small().to(device)

        checkpoint = torch.load(os.path.join(model_root, f'checkpoint-{epoch}.pth'), map_location='cpu', weights_only=False)
        if _is_slim_checkpoint(checkpoint, model_type):
            checkpoint_model = checkpoint['state_dict']
        else:
            checkpoint_model = checkpoint['model']
        msg = model.load_state_dict(checkpoint_model, strict=False)
    elif model_type == 'dino':
        from .dino.vision_transformer import vit_small, load_pretrained_weights
        if model_scale == 'vit_small':

            model = vit_small().to(device)
            load_pretrained_weights(
                model, 
                os.path.join(model_root, f'checkpoint{str(epoch).zfill(4)}.pth'), 
                'teacher', 'small', 16
            )
            msg = ''

        else:
            raise NotImplementedError
    elif model_type == 'ibot' or model_type == 'mugs':
        from .ibot.vision_transformer import vit_small, load_pretrained_weights
        if model_scale == 'vit_small':

            model = vit_small().to(device)
            load_pretrained_weights(
                model, 
                os.path.join(model_root, f'checkpoint{str(epoch).zfill(4)}.pth'), 
                'teacher', 'small', 16
            )
            msg = ''

        else:
            raise NotImplementedError
    
    elif model_type == 'ijepa':
        from .ijepa.deit import deit_base, deit_large

        if model_scale == 'vit_base':
            model = deit_base().to(device)
            checkpoint = torch.load(os.path.join(model_root, f'ijepa-ep{epoch}.pth.tar'), map_location=torch.device('cpu'), weights_only=False)
            if _is_slim_checkpoint(checkpoint, model_type):
                pretrained_dict = checkpoint['state_dict']
            else:
                pretrained_dict = checkpoint['encoder']
            pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
            msg = model.load_state_dict(pretrained_dict)

        else:
            raise NotImplementedError
    
    elif model_type == 'mocov3' or model_type == 'RINCE':
        from .mocov3.vit_moco import vit_small

        if model_scale == 'vit_small':
            model = vit_small().to(device)
            checkpoint = torch.load(os.path.join(model_root, f'checkpoint_{str(epoch).zfill(4)}.pth.tar'), map_location="cpu", weights_only=False)
            if _is_slim_checkpoint(checkpoint, model_type):
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint["state_dict"]
                state_dict = {k: v for k, v in state_dict.items() if k.startswith('module.momentum_encoder.')}
                state_dict = {k.replace("module.momentum_encoder.", ""): v for k, v in state_dict.items()}
            msg = model.load_state_dict(state_dict, strict=False)
        else:
            raise NotImplementedError

    elif model_type == 'mec':
        if model_scale == 'resnet50':
            from .resnet import resnet50
            model = resnet50().to(device)
            checkpoint = torch.load(os.path.join(model_root, f'checkpoint_{str(epoch).zfill(4)}.pth.tar'), map_location="cpu", weights_only=False)
            if _is_slim_checkpoint(checkpoint, model_type):
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint["state_dict"]
                state_dict = {k: v for k, v in state_dict.items() if k.startswith('module.encoder.')}
                state_dict = {k.replace("module.encoder.", ""): v for k, v in state_dict.items()}
            msg = model.load_state_dict(state_dict, strict=False)
        else:
            raise NotImplementedError

    elif model_type == 'resa':
        if model_scale == 'resnet50':
            from .resnet import resnet50
            model = resnet50().to(device)
            checkpoint = torch.load(os.path.join(model_root, f'checkpoint_{str(epoch).zfill(4)}.pth.tar'), map_location="cpu", weights_only=False)
            if _is_slim_checkpoint(checkpoint, model_type):
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint["state_dict"]
                state_dict = {k: v for k, v in state_dict.items() if k.startswith('module.encoder.')}
                state_dict = {k.replace("module.encoder.", ""): v for k, v in state_dict.items()}
            msg = model.load_state_dict(state_dict, strict=False)
        else:
            raise NotImplementedError

    elif model_type == 'simsiam':
        if model_scale == 'resnet50':
            from .resnet import resnet50
            model = resnet50().to(device)
            checkpoint = torch.load(os.path.join(model_root, f'checkpoint_{str(epoch).zfill(4)}.pth.tar'), map_location="cpu", weights_only=False)
            if _is_slim_checkpoint(checkpoint, model_type):
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint["state_dict"]
                state_dict = {k: v for k, v in state_dict.items() if k.startswith('module.encoder.')}
                state_dict = {k.replace("module.encoder.", ""): v for k, v in state_dict.items()}
            msg = model.load_state_dict(state_dict, strict=False)
            
        else:
            raise NotImplementedError

    elif model_type == 'densecl':
        if model_scale == 'resnet50':
            from .resnet import resnet50
            model = resnet50().to(device)
            checkpoint = torch.load(os.path.join(model_root, f'epoch_{epoch}.pth'), map_location="cpu", weights_only=False)
            if _is_slim_checkpoint(checkpoint, model_type):
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint["state_dict"]
                state_dict = {k: v for k, v in state_dict.items() if k.startswith('backbone.')}
                state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            msg = model.load_state_dict(state_dict, strict=False)
            
        else:
            raise NotImplementedError

    elif model_type == 'swav':
        if model_scale == 'resnet50':
            from .resnet import resnet50
            model = resnet50().to(device)
            checkpoint = torch.load(os.path.join(model_root, f'ckp-{epoch}.pth'), map_location="cpu", weights_only=False)
            if _is_slim_checkpoint(checkpoint, model_type):
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
                if "state_dict" in state_dict.keys():
                    state_dict = state_dict["state_dict"]
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            msg = model.load_state_dict(state_dict, strict=False)
            
        else:
            raise NotImplementedError
    
    elif model_type == 'vicreg' or model_type == 'vicregl':
        if model_scale == 'resnet50':
            from .resnet import resnet50
            model = resnet50().to(device)
            checkpoint = torch.load(os.path.join(model_root, f'model_{epoch}.pth'), map_location="cpu", weights_only=False)
            if _is_slim_checkpoint(checkpoint, model_type):
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint['model']
                if "state_dict" in state_dict.keys():
                    state_dict = state_dict["state_dict"]
                state_dict = {k: v for k, v in state_dict.items() if 'encoder' not in k}
                state_dict = {k.replace("module.backbone.", ""): v for k, v in state_dict.items()}
            msg = model.load_state_dict(state_dict, strict=False)
            
        else:
            raise NotImplementedError

    elif model_type == 'byol':
        from .byol.byol import EncoderwithProjection
        model = EncoderwithProjection().to(device)
        checkpoint = torch.load(os.path.join(model_root, f'resnet50_{str(epoch)}.pth.tar'), map_location="cpu", weights_only=False)
        if _is_slim_checkpoint(checkpoint, model_type):
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint['model']
            state_dict = {k: v for k, v in state_dict.items() if k.startswith('module.target_network.')}
            state_dict = {k.replace("module.target_network.", ""): v for k, v in state_dict.items()}

        msg = model.load_state_dict(state_dict, strict=False)
    
    elif model_type == 'barlowtwins':
        if model_scale == 'resnet50':
            from .resnet import resnet50
            model = resnet50().to(device)
            checkpoint = torch.load(os.path.join(model_root, f'checkpoint_{epoch}.pth'), map_location="cpu", weights_only=False)
            if _is_slim_checkpoint(checkpoint, model_type):
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint['model']
                if "state_dict" in state_dict.keys():
                    state_dict = state_dict["state_dict"]
                state_dict = {k: v for k, v in state_dict.items() if 'encoder' not in k}
                state_dict = {k.replace("module.backbone.", ""): v for k, v in state_dict.items()}
            msg = model.load_state_dict(state_dict, strict=False)
            
        else:
            raise NotImplementedError

    elif model_type == 'esvit':
        from .esvit.swin_transformer import SwinTransformer
        if model_scale == 'swint_tiny':
            model = SwinTransformer(
                img_size=[ 224, 224 ],
                in_chans=3,
                num_classes=1000,
                patch_size=4,
                embed_dim=96,
                depths=[ 2, 2, 6, 2 ],
                num_heads=[ 3, 6, 12, 24 ],
                window_size=7,
                mlp_ratio=4,
                qkv_bias=True,
                drop_rate=0,
                attn_drop_rate=0 ,
                drop_path_rate= 0.1,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                ape=False,
                patch_norm=True,
                use_dense_prediction=False,
            ).to(device)

            checkpoint = torch.load(os.path.join(model_root, f'checkpoint{str(epoch).zfill(4)}.pth'), map_location="cpu", weights_only=False)
            if _is_slim_checkpoint(checkpoint, model_type):
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint['teacher']
           
            msg = model.load_state_dict(state_dict, strict=False)
        else:
            raise NotImplementedError

    elif model_type == 'dinov2':
        from .dinov2.vision_transformer import vit_small as dinov2_vit_small
        if model_scale == 'vit_small':
            model = dinov2_vit_small(patch_size=14, num_register_tokens=4).to(device)
            ckpt_path = os.path.join(model_root, 'dinov2_vits14_reg4_pretrain.pth')
            checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            if _is_slim_checkpoint(checkpoint, model_type):
                state_dict = checkpoint['state_dict']
            else:
                # DINOv2 pretrained weights are already a bare encoder state_dict
                state_dict = checkpoint
            # DINOv2 was pretrained at 518x518; pos_embed size differs from 224x224.
            # The model interpolates pos_embed at runtime, so load the original size.
            if 'pos_embed' in state_dict:
                model.pos_embed = nn.Parameter(state_dict.pop('pos_embed'))
            msg = model.load_state_dict(state_dict, strict=False)
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError

    print(msg)
    return model
    
