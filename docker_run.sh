# how to run docker image
docker run -v $(pwd):/usr/src/app --gpus '"device=0"' -it first_test:latest /bin/bash