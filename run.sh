rm -rf time.dat
for i in `seq 100 100 2000`
do
  ./main.x 32 $i > out
  time=`grep "avg time per matmul" out`
  echo $i '     ' $time '   ' >> time_vs_nmat.dat
done
