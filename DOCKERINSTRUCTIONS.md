## INSTRUCTIONS: how to run code in a docker container on the upenn server:

#### ensure you are on the upenn vpn (GlobalProtect)

#### ssh into the server with:

```
ssh bhavyasp@mac.xlab.upenn.edu
```

#### cd into project

```
cd ~/wind-farm-control
```

#### pull changes to code

```
git pull
```

#### make checkpoint directory (if already made, will ignore, if not it will make it.) you can also check it by using ls.

```

mkdir -p ~/windfarm_checkpoints

ls ~
```

#### set configuration

```
docker run -d \
  --cpus=16 \
  --name windfarm_train \
  windfarm
```

#### make sure you're logged into wandb

```
wandb login
```

#### check if wandb api key is environment variable, if not, add.

```
echo $WANDB_API_KEY

export WANDB_API_KEY=<your_api_key>
```

#### rebuild docker

```
docker build --no-cache -t windfarm .
```

#### check if old container exists. if so, remove

```
docker ps -a

docker rm -f windfarm_train
```

#### set configuration

```
docker run -d \
    --cpus=32 \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    -v ~/windfarm_checkpoints:/app/checkpoints \
    --name windfarm_train \
    windfarm \
    python scripts/torchrl_mappo/train_mappo_floris.py \
        --checkpoint-dir checkpoints
```

#### check if it started/is still running

```
docker ps
```

#### logs (should also be in wandb)

```
docker logs -f windfarm_train
```

#### check checkpoints:

```
ls ~/windfarm_checkpoints
```

#### exit (training still continues since docker runs detached)

```
exit
```

#### stop training:

```
docker stop windfarm_train
```

#### resume training:

```
docker rm windfarm_train

docker run -d \
    --cpus=16 \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    -v ~/windfarm_checkpoints:/app/checkpoints \
    --name windfarm_train \
    windfarm \
    python scripts/torchrl_mappo/train_mappo_floris.py \
        --resume checkpoints/latest.pt


IF YOU WANT TO CHANGE NUM ITERS OR ANYTHING, CAN ADD ARGS LIKE THIS:

--resume checkpoints/latest.pt \
--n-iters 6000
```
