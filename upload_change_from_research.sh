cp -r ~/Documents/omni_research/data      ~/Documents/omni_torch
cp -r ~/Documents/omni_research/options   ~/Documents/omni_torch
cp -r ~/Documents/omni_research/visualize ~/Documents/omni_torch
cp ~/Documents/omni_research/networks/*.py ~/Documents/omni_torch/networks
git config core.autocrlf true
git add .
git commit -m "start"
git push
