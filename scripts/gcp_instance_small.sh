export IMAGE_FAMILY="pytorch-latest-gpu" # pytorch-latest-cu100
export ZONE="us-east1-c"
export INSTANCE_NAME="instance-1"
export INSTANCE_TYPE="n1-highmem-8"

gcloud compute instances create $INSTANCE_NAME \
        --zone=$ZONE \
        #--image c2-deeplearning-pytorch-1-3-cu100-20191213
        --image-family=$IMAGE_FAMILY \
        --image-project=deeplearning-platform-release \ #ml-images
        --maintenance-policy=TERMINATE \
        --accelerator="type=nvidia-tesla-k80,count=1" \   #nvidia-tesla-k80, nvidia-tesla-p4
        --machine-type=$INSTANCE_TYPE \
        --boot-disk-size=100GB \
        --metadata="install-nvidia-driver=True" \
        --preemptible


        --image=c2-deeplearning-pytorch-1-3-cu100-20191213