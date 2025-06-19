# Transformer Layers as Painters 🧑‍🎨
<p align="center">
  📚 <a href="https://arxiv.org/abs/2407.09298">[Paper]</a>
</p>

## Requirements
1. Please run the following commands to set up the basic python environment. The
   repository can be cloned via HTTPS and does not require the separate private
   repository mentioned in early versions of this README.
```
git clone https://github.com/floatingbigcat/transformer_layers_as_painters.git
cd transformer_layers_as_painters

# We use python 3.10

python -m venv painter_env
source painter_env/bin/activate
pip install -r requirements.txt
```

2. Our evaluation of GPT style model is based on [lm_eval](https://github.com/EleutherAI/lm-evaluation-harness), we fix the module commit to to make sure our experiments is reproducible, and make minmal modification on `llama` and `mistral` to enable our methods on GPT style model.
Please run the following commands to install lm_eval on the current python environment.
```
cd gpt

cd lm-eval
git submodule update --init

cp -f ../__main__.py lm_eval/
cp -f ../evaluator.py lm_eval/
cp -f ../modify_model.py lm_eval/
cp -f ../routing_llama.py lm_eval/
cp -f ../routing_mistral.py lm_eval/
cp -f ../routing_neox.py lm_eval/

pip install -e .
```

## Example Usage
Basically, you can run all our methods by simply editing the arguments in the
`example.sh` scripts under `gpt/` or `bert/`. If you do not have a GPU
available, set `CUDA_VISIBLE_DEVICES=""` and pass `--device cpu` to `lm_eval`
as shown below.

### GPT 
```
cd gpt
export CUDA_VISIBLE_DEVICES=
bash example.sh --device cpu
```

### Bert
```
cd bert
bash example.sh --device cpu
```

## Cosine Similiary Plot

please check `./cos_sim_plotter.ipynb` about how we obtain the cosine similiary heat map of hidden states over layers

