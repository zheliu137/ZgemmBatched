
set xlabel 'number of (32x32) matrix'
set ylabel 'average time per matrix'

plot 'out' u 1:2 w l title 'zgemmbatched', '../batched_Zgemm_perf/test.dat' w l title 'zgemmstridedbatched'
