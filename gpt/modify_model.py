import types
from lm_eval.routing_llama import LlamaSdpaAttention, LlamaDecoderLayer, LlamaModel


def modify_llama(lm, args=None):
    # Add new attribute to llama model
    lm.model.model.method = args.method
    lm.model.model.start_layer = args.start_layer
    lm.model.model.repeat_time_or_seed = args.repeat_time_or_seed
    lm.model.model.hidden_state_folder_path = args.hidden_state_folder_path

    # Modify the forward function into run different layer
    lm.model.model.forward = types.MethodType(LlamaModel.forward, lm.model.model)
    # lm.model.model._pass_through_layer = types.MethodType(
    #     LlamaModel._pass_through_layer, lm.model.model
    # )

    for i in range(len(lm.model.model.layers)):
        lm.model.model.layers[i].forward = types.MethodType(
            LlamaDecoderLayer.forward, lm.model.model.layers[i]
        )

    for i in range(len(lm.model.model.layers)):
        lm.model.model.layers[i].self_attn.forward = types.MethodType(
            LlamaSdpaAttention.forward, lm.model.model.layers[i].self_attn
        )

    return lm
