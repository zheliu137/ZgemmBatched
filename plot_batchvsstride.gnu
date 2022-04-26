
set xlabel 'number of (32x32) matrix'
set ylabel 'average time per matrix'

plot 'time_vs_nmat.dat' u 1:2 w l title 'zgemmbatched', 'strided_TimeVsNumber.dat' w l title 'zgemmstridedbatched'
