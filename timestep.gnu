
set xlabel 'matrix row/column N'
set ylabel 'average time per matrix'

plot 'time.dat' u 1:2 w l title 'zgemmbatched', '../batched_Zgemm_perf/time.dat' w l title 'zgemmstridedbatched'
