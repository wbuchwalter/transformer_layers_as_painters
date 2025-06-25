import types
from lm_eval.modeling_llama import LlamaAttention, LlamaDecoderLayer, LlamaCustomModel
from lm_eval.modeling_gemma import CuGemma3ForCausalLM, Gemma3DecoderLayer


def modify_gemma(lm, args=None):
    print(f"modifying: {lm.model.model.__class__.__name__}")
    # breakpoint()
    # lm.model.model.forward = types.MethodType(
    #     CuGemma3ForCausalLM.forward, lm.model.model
    # )
    #'Gemma3TextModel' object has no attribute 'language_model'
    # AttributeError: 'Gemma3Model' object has no attribute 'layers'

    layers = None
    if lm.model.model.__class__.__name__ == "Gemma3Model":
        layers = lm.model.model.language_model.layers
    elif lm.model.model.__class__.__name__ == "Gemma3TextModel":
        layers = lm.model.model.layers

    for i in range(len(layers)):
        layers[i].forward = types.MethodType(Gemma3DecoderLayer.forward, layers[i])
        layers[i].repeat_mlp = 1

    if args.method == "smart":
        layers[8].repeat_mlp = 5
        layers[9].repeat_mlp = 5
        layers[10].repeat_mlp = 5

    return lm


def modify_llama(lm, args=None):
    # Add new attribute to llama model
    # lm.model.model.start_layer = args.start_layer
    # lm.model.model.repeat_time_or_seed = args.repeat_time_or_seed
    # lm.model.model.hidden_state_folder_path = args.hidden_state_folder_path

    # Modify the forward function into run different layer
    lm.model.model.set_method = types.MethodType(
        LlamaCustomModel.set_method, lm.model.model
    )
    lm.model.model.forward = types.MethodType(LlamaCustomModel.forward, lm.model.model)
    lm.model.model.set_method(args.method)

    # lm.model.model._pass_through_layer = types.MethodType(
    #     LlamaModel._pass_through_layer, lm.model.model
    # )

    for i in range(len(lm.model.model.layers)):
        lm.model.model.layers[i].forward = types.MethodType(
            LlamaDecoderLayer.forward, lm.model.model.layers[i]
        )

    for i in range(len(lm.model.model.layers)):
        lm.model.model.layers[i].self_attn.forward = types.MethodType(
            LlamaAttention.forward, lm.model.model.layers[i].self_attn
        )

    return lm
