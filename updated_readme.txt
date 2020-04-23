git clone https://github.com/openai/gpt-2.git && cd gpt-2
pip3 install fire
pip3 install tensorflow==1.14.0
pip3 install -r requirements.txt
sed -i 's/top_k=0/top_k=40/g' src/interactive_conditional_samples.py

#choose one
"""
python3 download_model.py 124M
python3 download_model.py 355M
python3 download_model.py 774M
python3 download_model.py 1558M
"""
python3 src/interactive_conditional_samples.py

if any error follow blog to reinstall and recode

https://lambdalabs.com/blog/run-openais-new-gpt-2-text-generator-code-with-your-gpu/