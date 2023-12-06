git clone https://github.com/yuwenmichael/Grounding-DINO-Batch-Inference.git
mv Grounding-DINO-Batch-Inference/GroundingDINO .
cd GroundingDINO
pip install -q -e .
mkdir weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

cd ..
git clone https://github.com/IDEA-Research/Grounded-Segment-Anything
rm -r Grounded-Segment-Anything/GroundingDINO
mv GroundingDINO Grounded-Segment-Anything

cd Grounded-Segment-Anything
pip install -q -r requirements.txt
cd segment_anything
rm -r *
git clone https://github.com/SysCV/sam-hq.git
shopt -s dotglob && mv -t . sam-hq/*
pip install -q .
cd ..

wget https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth
