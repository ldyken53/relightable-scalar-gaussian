#!/bin/bash
set -o errexit # Exit on error
set -o nounset # Trigger error when expanding unset variables


dataset="combustion"

root_dir=./output/${dataset}
list=$(basename -a $root_dir/*/)
echo "TFs list: " $list

for i in $list; do
echo "Rendering $i"
#todo: view arguments maybe need to change if you use customize views
#* generate images
python3 render.py -vo ./Data/combustion/${i} \
                  -so ${root_dir}/${i}/neilf \
                  --output ${root_dir}/${i}/neilf \
                  --useHeadlight

# * only evaluation, test rendering time
python3 render.py -vo ./Data/combustion/${i} \
                  -so ${root_dir}/${i}/neilf \
                  --output ${root_dir}/${i}/neilf \
                  --useHeadlight \
                  --EvalTime > ${root_dir}/${i}/neilf/eval_time.txt
done