git pull
cp -r ~/Documents/omni_torch/data       ~/Documents/omni_research
cp -r ~/Documents/omni_torch/options    ~/Documents/omni_research
cp -r ~/Documents/omni_torch/visualize  ~/Documents/omni_research
cp ~/Documents/omni_torch/networks/*.py ~/Documents/omni_research/networks
cd ~/Documents/omni_research
git config core.autocrlf true
git add .
git commit -m "research"
git push

