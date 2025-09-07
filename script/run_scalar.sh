#!/bin/bash
set -o errexit # Exit on error
set -o nounset # Trigger error when expanding unset variables

root_dir=$1
exp_name=$2
echo "Dataset root dir: ${root_dir}"

# Collect indices from folders named TF* and NSTF* (assumes both sets are present for each index)
# Find all folders in the root_dir
indices=()
for folder in "$root_dir"/TF*/ "$root_dir"/NSTF*/; do
    base=$(basename "$folder")
    # Extract 2 digits at the end of the folder name (matches both TF01 and NSTF01)
    idx=$(echo "$base" | grep -oE '[0-9]{2}$')
    if [[ -n "$idx" && ! " ${indices[*]} " =~ " $idx " ]]; then
        indices+=("$idx")
    fi
done

echo "Found indices: ${indices[*]}"

for i in "${indices[@]}"; do
    echo "Processing index: ${i}"
    tic=$(date +%s)
    
    # First command with NSTF{i}
    python train.py --eval \
        -s "${root_dir}/NSTF${i}" \
        -m "output/${exp_name}scalar/TF${i}/3dgs" \
        --lambda_normal_render_depth 0.01 \
        --lambda_opacity 0.1 \
        --densification_interval 500 \
        --densify_grad_normal_threshold 0.000004 \
        --save_training_vis \
        --is_scalar
    
    # Second command with TF{i}
    python train.py --eval \
        -s "${root_dir}/TF${i}" \
        -m "output/${exp_name}scalar/TF${i}/neilf" \
        -c "output/${exp_name}scalar/TF${i}/3dgs/chkpnt30000.pth" \
        -t phong \
        --lambda_normal_render_depth 0.01 \
        --lambda_opacity 0.1 \
        --lambda_phong 1.0 \
        --densify_until_iter 32000 \
        --lambda_render 0.0 \
        --use_global_shs \
        --finetune_visibility \
        --iterations 40000 \
        --test_interval 1000 \
        --checkpoint_interval 2500 \
        --lambda_offset_color_sparsity 0.01 \
        --lambda_ambient_factor_smooth 0.01 \
        --lambda_specular_factor_smooth 0.01 \
        --lambda_normal_smooth 0.00 \
        --lambda_diffuse_factor_smooth 0.01 \
        --save_training_vis \
        --is_scalar

    toc=$(date +%s)
    echo "Processing index ${i} took $((toc - tic)) seconds" >> "output/${exp_name}scalar/TF${i}/time.txt"
done
