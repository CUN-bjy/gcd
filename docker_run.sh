# how to run docker image
docker run -v $(pwd):/usr/src/app -v $(pwd)/download_resources:/root/.cache/torch/hub/ --name Spark_gcd --ipc=host --gpus '"device=1"' --rm -it spark_gcd:latest /bin/bash