import jclip as clip


def load_clip(freeze_version=0):
    """freeze part of CLIP model.

    Args:
        freeze_version (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    assert freeze_version in [0, 1, 2, 3, 4, 5, 6]
    model, preprocess = clip.load("pretrained/ViT-B-32.pkl")
    model.train()
    
    def _freeze_version_1():
        """冻结全部文本层"""
        model.token_embedding.weight.requires_grad = False
        model.positional_embedding.requires_grad = False
        model.text_projection.requires_grad = False
        model.ln_final.weight.requires_grad = False
        model.ln_final.bias.requires_grad = False
        for param in model.transformer.parameters():
            param.requires_grad = False
    
    def _freeze_version_2():
        """冻结全部图像层"""
        for param in model.visual.parameters():
            param.requires_grad = False
       
    def _freeze_version_3():
        """冻结浅层图像层"""
        _freeze_version_2()
        unfreeze_layers = ['resblocks.9', 'resblocks.10', 'resblocks.11', 'ln_post']
        for name, param in model.visual.named_parameters():
            if any(layer in name for layer in unfreeze_layers):
                param.requires_grad = True
            else:
                param.requires_grad = False
    
    def _freeze_version_4():
        """冻结高层图像层"""
        _freeze_version_2()
        unfreeze_layers = ['resblocks.0', 'resblocks.1', 'resblocks.2']
        for name, param in model.visual.named_parameters():
            if any(layer in name for layer in unfreeze_layers):
                param.requires_grad = True
            else:
                param.requires_grad = False
                
        model.visual.class_embedding.requires_grad = True
        model.visual.positional_embedding.requires_grad = True
        model.visual.proj.requires_grad = True
        model.visual.conv1.weight.requires_grad = True
        model.visual.ln_pre.weight.requires_grad = True
        model.visual.ln_pre.bias.requires_grad = True
        
    def _freeze_version_5():
        """冻结浅层文本层"""
        _freeze_version_1()
        unfreeze_layers = ['resblocks.9', 'resblocks.10', 'resblocks.11', 'ln_final']
        for name, param in model.transformer.named_parameters():
            if any(layer in name for layer in unfreeze_layers):
                param.requires_grad = True
            else:
                param.requires_grad = False
                
    def _freeze_version_6():
        """冻结高层文本层"""
        _freeze_version_1()
        model.token_embedding.weight.requires_grad = True
        model.positional_embedding.requires_grad = True
        model.text_projection.requires_grad = True
        model.ln_final.weight.requires_grad = True
        model.ln_final.bias.requires_grad = True
        unfreeze_layers = ['resblocks.0', 'resblocks.1', 'resblocks.2']
        for name, param in model.transformer.named_parameters():
            if any(layer in name for layer in unfreeze_layers):
                param.requires_grad = True
            else:
                param.requires_grad = False
                
    freeze_functions = {
        0: lambda: print('NO Freeze'),
        1: _freeze_version_1,
        2: _freeze_version_2,
        3: _freeze_version_3,
        4: _freeze_version_4,
        5: _freeze_version_5,
        6: _freeze_version_6
    }
    freeze_functions[freeze_version]()
    return model, preprocess


if __name__ == '__main__':
    model, preprocess = load_clip(freeze_version=0)
    
    for name, param in model.named_parameters():
        print(name.ljust(60, '-'), param.requires_grad)
    # for name, param in model.named_parameters():
    #     print(name.ljust(60, '-'), param.requires_grad)
    
    


