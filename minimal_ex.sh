#quickstart script
echo "Downloading Data"
wget -O test_data http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Amazon_Instant_Video_5.json.gz
echo "Preparing Data"
python prepare_data.py --create-emb --epochs 2 test_data prepared_data
echo "Learning net"
python main.py prepared_data --emb prepared_data_emb.txt