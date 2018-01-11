#quickstart script
echo "Downloading Data"
wget -O test_data http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Amazon_Instant_Video_5.json.gz
echo "Preparing Data"
python prepare_data.py test_data prepared_data
echo "Learning net"
python han.py prepared_data --cuda