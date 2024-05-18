# OpenSoraTraining
A hacker's guide to training Open Sora Plan on your custom dataset and GPUs.

### Setting up Vast for training

To SSH into a Vast.ai instance, you need to follow these general steps:

1) Generate and add your SSH key to Vast

The command bellow will generate an SSH key and store it in the ```~/.ssh``` with your desired filename.

```
ssh-keygen -t rsa -b 4096 -C "your_email@example.com" -f ~/.ssh/<your_filename>.pem
```

When you ls into ```~/.ssh```  you will see the private and public key ```.pem``` and ```.pem.pub``` respectively.
Now go to the Vast dashboard and add your ssh key. Copy ur ssh key with the command and add your ssh key to vast:

```
pbcopy < ~/.ssh/your_file.pem.pub
```

2) Pay hommage to Vast

Go to the [billing](https://cloud.vast.ai/billing/), input your credit card info, get some free credits blah blah.

3) Creating a template on Vast

On Vast.ai, a template is a pre-configured environment that includes specific software and settings needed for a particular task. 

Templates can include (saves time):

Specific versions of software (e.g., PyTorch, TensorFlow).
Runtime environments with specific libraries and dependencies.
Custom settings and configurations tailored to certain workflows or applications.


Since you're working with a machine learning model, the "Pytorch 2.2.0 Cuda12.1 Devel" template or any other PyTorch template with the appropriate CUDA support should be suitable. These templates will have the necessary libraries pre-installed and allow you to set up your environment without Docker.


3) Renting out GPUs

Now let's [find a GPU](https://cloud.vast.ai/) to rent out! 
When training a video generation model we're looking for something like somewhere in the 80gb of memory range. Most of the GPUs from the past 5 years (A100s, Tesla GPUs, H100s they all work well just a matter of speed).

4) Your instance is running! SSH now!

Go click on the "Open SSH Interface" button and click on the Add SSH Key. Now your SSH key is added to the instance,
and you can SSH in!

```
ssh -i ~/.ssh/your_file.pem -p <port_number> root@<instance_up> -L 8080:localhost:8080
```

Wait a minute and then enter your paraphrase...

5) Clone Open Sora Plan into your Vast instance

```
git clone https://github.com/RaccoonResearch/Open-Sora-Plan.git
git checkout raccoon/training
```

if you do ls you will see Open-Sora-Plan is indeed in your directory. Now ```cd Open-Sora-Plan```

6) Install Open Sora Plan packages

Make sure you have [conda](https://www.anaconda.com/download) installed.
```
conda --version
```

Then install required packages in this order:
```
cd Open-Sora-Plan
conda create -n opensora python=3.8 -y
conda activate opensora
pip install -e .
```

Install additional packages for training cases:
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

Install optional requirements such as static type checking:
```
pip install -e '.[dev]'
```

Now let's install the weights for the model from  [Hugging face](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.0.0/tree/main?clone=true). Clone the repo.

Make sure we have git lfs installed. Git lsf is an extension to git and it's used to handle large files more efficiently.
```
sudo apt install git-lfs
git lfs install
git clone https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.0.0
```

### Training Open-Sora-Plan

1) Take a look at  [Open Sora Plan](https://github.com/RaccoonResearch/Open-Sora-Plan)

More specifically take a look at [Raccoon Research's changes](https://github.com/RaccoonResearch/Open-Sora-Plan/pull/1/files)

```
# BEFORE: 
num_processes: 8
gpu_ids: 0,1,2,3,4,5,6,7
use_cpu: false


# AFTER (yes you need that unnecessary comma after the 0):
num_processes: 1
gpu_ids: 0,
use_cpu: false
```

We're going to be training on 1 GPU so we're going to set our gpu_ids count to be 0 and the use_cpu to false (unless you want to run this on your PC hehe) 

```
# BEFORE: 
--ae_path CausalVAEModel_4x8x8 \
--data_path /remote-home1/dataset/sharegpt4v_path_cap_.json \
--video_folder /remote-home1/dataset/data_split_tt \

# AFTER:
--ae_path "./Open-Sora-Plan-v1.0.0/vae" \
--data_path "./videos/captions.json" \
--video_folder "./videos" \
```

We're going to change our model path to be ```./Open-Sora-Plan-v1.0.0/vae```. We're also going to change our 
```data_path``` and ```video_folder``` paths which will contain our captions and mp4 videos perpared.

```
# BEFORE: 
--dataloader_num_workers 10 \

# AFTER: 
--dataloader_num_workers 1 \
```

We're going to set a single process handles all data loading.

2) Back to Vast

```
cd Open-Sora-Plan
mkdir videos
```

3) Your video dataset

You can use scp to transfer it to the Vast.ai instance

```
scp -i ~/.ssh/your_private.pem -P <port> ./renders/*.mp4 root@<instance_ip>:/root/Open-Sora-Plan/videos/
```

Generate captions for your videos
```
python3 scripts/combinations_to_captions.py --json_file combinations.json --video_folder renders
```

Add that to the videos folder too (MAKE SURE THE PATHS MATCH WHAT'S ON YOUR VAST INSTANCE):
```
scp -i ~/.ssh/your_private.pem -P <port> ./captions.json root@<instance_ip>:/root/Open-Sora-Plan/videos/
```

### NOW TRAIN

Go get your API key from [wandb](https://wandb.ai/home)

```
WANDB_KEY=your_key bash scripts/text_condition/train_videoae_65x512x512.sh
```
