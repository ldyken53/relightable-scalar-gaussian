#!/bin/bash
set -o errexit # Exit on error
set -o nounset # Trigger error when expanding unset variables


dataset="chameleon"

root_dir=./output/allinit
data_dir=./Data/${dataset}
list=$(basename -a $data_dir/TF*/)
echo "TFs list: " $list

# for i in $list; do
# echo "Rendering $i"
# python3 render.py -vo ./Data/${dataset}/${i} \
#                   -so ${root_dir}/neilf \
#                   --output ${root_dir}/${i}/eval \
#                   --useHeadlight --is_scalar

# # * only evaluation, test rendering time
# # python3 render.py -vo ./Data/combustion/${i} \
# #                   -so ${root_dir}/${i}/neilf \
# #                   --output ${root_dir}/${i}/neilf \
# #                   --useHeadlight \
# #                   --EvalTime > ${root_dir}/${i}/neilf/eval_time.txt
# done

indices=()
for folder in "$data_dir"/TF*/; do
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
    python3 render.py -vo ./Data/${dataset}/ \
                  -so ${root_dir} \
                  --output ${root_dir}/eval \
                  --useHeadlight --is_scalar
done