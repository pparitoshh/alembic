CPU: Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz · 6 threads · llama.cpp 46847e615 · context 8192
3 pass(es), mean ± std; machine in normal use (1-min load average per pass: 12.57, 6.89, 6.21)

| model | file GB | peak RAM GB (max) | pp512 tok/s | tg128 tok/s @0 | tg128 tok/s @2048 |
|---|---|---|---|---|---|
| Q3_K_M | 2.08 | 3.81 | 49.4 ± 4.1 | 12.6 ± 0.9 | 7.6 ± 0.4 |
| Q4_K_M | 2.50 | 5.15 | 50.9 ± 2.5 | 11.3 ± 0.7 | 7.2 ± 0.4 |
| Q5_K_M | 2.89 | 3.88 | 47.6 ± 2.6 | 9.6 ± 0.6 | 6.5 ± 0.5 |
| Q6_K | 3.31 | 4.27 | 48.9 ± 3.7 | 8.8 ± 0.7 | 6.1 ± 0.5 |
| Q8_0 | 4.28 | 5.18 | 38.4 ± 1.9 | 7.0 ± 0.5 | 5.2 ± 0.3 |
