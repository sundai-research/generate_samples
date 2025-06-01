# install

```shell=
pyenv install 3.12.0
pyenv virtualenv 3.12 sglang-env
pyenv local sglang-env

```

# serve a model

```shell=
export model_path="Qwen/Qwen3-1.7B"
# In bash you evaluate an env var by prefixing it with $ (or ${…})
python -m sglang_router.launch_server --model-path "${model_path}" --dp-size 4 --reasoning-parser qwen3 --host 0.0.0.0
```

# do generation
