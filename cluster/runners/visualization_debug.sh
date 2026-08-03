cd /mozaik

echo "Waiting for debugger to attach on port 5678..."
# python devtools/visualization_demo.py 
pip list
python -m debugpy --listen 0.0.0.0:5678 --wait-for-client devtools/visualization_demo.py 