# W3A16
export HF_ENDPOINT="https://hf-mirror.com"
# export HF_HOME="/home/public/shared_hf_cache"
gpu_no=$1
sparsity_ratio=$2
method=$3
model_name=$4
wbits=$5
abits=$6

de=${de:-""}

cmd="CUDA_VISIBLE_DEVICES=$gpu_no python"
if [[ "$de" == "de" ]]; then
    cmd="$cmd -m debugpy --listen 7878 --wait-for-client"
fi
model_path=/home/public/shared_hf_cache/$model_name
cmd="$cmd main.py \

--model $model_name \
--epochs 0 --output_dir ./log/$method-$model_name-w$wbits-a$abits-$sparsity_type/ \
--eval_ppl --wbits $wbits --abits $abits \
--tasks arc_easy,arc_challenge,piqa,winogrande \
--sparsity_ratio $sparsity_ratio \
--prune_method $method \
--sparsity_type unstructured"

echo $cmd
eval $cmd
