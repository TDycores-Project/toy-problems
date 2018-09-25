echo "p = 1-x"
python study.py 0.0 0

echo "p = (x-1/2)^2"
python study.py 0.0 1

echo "p = (1-4)^4 + (1-y)^3 (1-x) + sin(1-y)*cos(1-x)"
python study.py 0.0 2


echo "p = 1-x"
python study.py 1.0 0

echo "p = (x-1/2)^2"
python study.py 1.0 1

echo "p = (1-4)^4 + (1-y)^3 (1-x) + sin(1-y)*cos(1-x)"
python study.py 1.0 2
