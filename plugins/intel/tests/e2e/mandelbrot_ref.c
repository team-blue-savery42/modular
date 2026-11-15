//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Sean Avery. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

// CPU float64 reference for MAX's test_mandelbrot: the sum of the
// 4,096 x 4,096 iteration counts with fused multiply-add used in four ways,
// next to the B70's sum (tests/e2e/mandelbrot.mojo). The constants are the
// doubles in the kernel's SPIR-V (scale_x = 2.47 / 4096, scale_y = 2.24 /
// 4096). The loop follows the test's: lanes of `simd width` (2 on the B70)
// iterate until all of them have left the set, and a lane's count grows
// while its |z|^2 <= 4.
// Build and run (any x86-64 host with FMA in libm):
//   clang -O2 -ffp-contract=off -fopenmp mandelbrot_ref.c -o mandelbrot_ref -lm
//   ./mandelbrot_ref [simd width]
// Measured (widths 1 and 2 alike):
//   loop fmas as written, cx/cy mul then add (the B70's SPIR-V): 4687759364
//   loop fmas as written, cx/cy fused (NVIDIA's PTX):           4687767697
//   no fma anywhere:                                            4687754845
//   loop without fma, cx/cy fused:                              4687764907
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
enum { W = 4096, H = 4096, MAX_ITERS = 1000 };
static const double min_x = -2.0, min_y = -1.12;
static const double scale_x = 0.00060302734375000005; /* (0.47 - -2.0) / 4096 */
static const double scale_y =
    0.00054687500000000005; /* (1.12 - -1.12) / 4096 */
/* The summed counts of pixels (row, col .. col + width - 1). fc: cx and cy
   with one fma (else a multiply, then an add); fl: the loop's fmas as
   ComplexSIMD writes them (else each a multiply, then an add). */
static long long pixels(long row, long col, int fc, int fl, int width) {
  double cx[2], cy, re[2] = {0, 0}, im[2] = {0, 0};
  long long it[2] = {0, 0};
  int in[2] = {1, 1};
  cy = fc ? fma((double)row, scale_y, min_y) : (double)row * scale_y + min_y;
  for (int l = 0; l < width; l++) {
    double x = (double)col + (double)l;
    cx[l] = fc ? fma(x, scale_x, min_x) : x * scale_x + min_x;
  }
  for (int k = 0; k < MAX_ITERS; k++) {
    int any = 0;
    for (int l = 0; l < width; l++)
      any |= in[l];
    if (!any)
      break;
    for (int l = 0; l < width; l++) {
      double r = re[l], i = im[l], n;
      if (fl)
        n = fma(r, r, i * i);
      else
        n = r * r + i * i;
      in[l] = n <= 4.0;
      it[l] += in[l];
      if (fl) {
        re[l] = fma(r, r, fma(i, -i, cx[l]));
        im[l] = fma(r, i + i, cy);
      } else {
        re[l] = r * r + (-i * i + cx[l]);
        im[l] = r * (i + i) + cy;
      }
    }
  }
  long long s = 0;
  for (int l = 0; l < width; l++)
    s += it[l];
  return s;
}
int main(int argc, char **argv) {
  int width = argc > 1 ? atoi(argv[1]) : 2;
  if (width != 1 && width != 2) {
    fprintf(stderr, "usage: mandelbrot_ref [1|2]\n");
    return 2;
  }
  const char *names[] = {"loop fmas as written, cx/cy mul then add (the B70)",
                         "loop fmas as written, cx/cy fused (NVIDIA)",
                         "no fma anywhere", "loop without fma, cx/cy fused"};
  int fcs[] = {0, 1, 0, 1}, fls[] = {1, 1, 0, 0};
  for (int v = 0; v < 4; v++) {
    long long total = 0;
#pragma omp parallel for reduction(+ : total) schedule(dynamic, 16)
    for (long row = 0; row < H; row++)
      for (long col = 0; col < W; col += width)
        total += pixels(row, col, fcs[v], fls[v], width);
    printf("simd width %d, %s: %lld\n", width, names[v], total);
    fflush(stdout);
  }
  return 0;
}
