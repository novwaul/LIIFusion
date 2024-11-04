echo 'celebahq-x5.3' &&
python test.py --config ./configs/test/test-celebAHQ-5.3.yaml --model $1 --gpu $2 &&
echo 'celebahq-x7' &&
python test.py --config ./configs/test/test-celebAHQ-7.yaml --model $1 --gpu $2 &&
echo 'celebahq-x10' &&
python test.py --config ./configs/test/test-celebAHQ-10.yaml --model $1 --gpu $2 &&

echo 'celebahq-x10.7' &&
python test.py --config ./configs/test/test-celebAHQ-10.7.yaml --model $1 --gpu $2 &&
echo 'celebahq-x12' &&
python test.py --config ./configs/test/test-celebAHQ-12.yaml --model $1 --gpu $2 &&

true
