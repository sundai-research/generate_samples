# install

```shell=
conda create -n sglang python=3.12 -y
conda activate sglang
pip install sglang-router
```

# serve a model

```shell=
export model_path="Qwen/Qwen3-1.7B"
python -m sglang_router.launch_server --model-path $model_path --dp-size 4 --reasoning-parser qwen3 --host 0.0.0.0
```

# 