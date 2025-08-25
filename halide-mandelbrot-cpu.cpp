#include "Halide.h"
#include <vector>

using namespace Halide;

int main(int argc, char **argv) {
    // Parametri di input
    Param<int> width("width");
    Param<int> height("height");
    Param<float> xmin("xmin");
    Param<float> xmax("xmax");
    Param<float> ymin("ymin");
    Param<float> ymax("ymax");
    Param<int> max_iter("max_iter");
    
    Var x("x"), y("y"), c("c");
    Var xi("xi"), yi("yi"), xo("xo"), yo("yo");
    Var xii("xii"), yii("yii");
    
    // Tabella colori per la visualizzazione
    ImageParam color_lut(UInt(8), 2, "color_lut"); // [3 x NCOLORI]
    Param<int> ncolors("ncolors");
    
    // Pre-calcola la trasformazione di coordinate pixel -> piano complesso
    Func coords("coords");
    coords(x, y) = Tuple(
        xmin + (xmax - xmin) * cast<float>(x) / (width - 1),  // parte reale
        ymax - (ymax - ymin) * cast<float>(y) / (height - 1)  // parte immaginaria
    );
    
    // Calcolo delle iterazioni Mandelbrot
    Func iteration("iteration");
    Expr re = coords(x, y)[0];
    Expr im = coords(x, y)[1];
    
    // Inizializza z = 0, contatore iterazioni = 0
    Expr zr = 0.0f, zi = 0.0f;
    Expr iter_count = 0;
    
    const int MAX_UNROLL_ITER = 80;
    
    // Srotolamento manuale del loop in blocchi di 4 per migliorare la vettorizzazione
    for (int block = 0; block < MAX_UNROLL_ITER/4; block++) {
        for (int i = 0; i < 4; i++) {
            Expr zr_sq = zr * zr;
            Expr zi_sq = zi * zi;
            Expr magnitude_sq = zr_sq + zi_sq;
            Expr still_bounded = magnitude_sq <= 4.0f;
            
            // Formula Mandelbrot: z = z² + c
            Expr zr_new = zr_sq - zi_sq + re;
            Expr zi_new = 2.0f * zr * zi + im;
            
            // Aggiorna solo se il punto è ancora limitato
            zr = select(still_bounded, zr_new, zr);
            zi = select(still_bounded, zi_new, zi);
            iter_count = select(still_bounded, iter_count + 1, iter_count);
        }
    }
    
    iteration(x, y) = cast<int32_t>(iter_count);
    
    // Mappatura colori: nero per punti nell'insieme, tabella colori per gli altri
    Func color_mapping("color_mapping");
    Expr iter_val = iteration(x, y);
    Expr in_set = iter_val >= MAX_UNROLL_ITER;
    Expr color_idx = clamp(iter_val % ncolors, 0, ncolors - 1);
    
    color_mapping(x, y, c) = cast<uint8_t>(
        select(in_set, 0, color_lut(c, color_idx))
    );
    
    Func mandelbrot("mandelbrot");
    mandelbrot(x, y, c) = color_mapping(x, y, c);
    
    // Target CPU con ottimizzazioni SIMD
    Target target = get_host_target();
    target = target.with_feature(Target::SSE41)
                   .with_feature(Target::AVX);
    
    if (target.has_feature(Target::AVX2)) {
        target = target.with_feature(Target::AVX2);
    }
    
    // Tiling gerarchico per efficienza cache
    const int outer_tile_x = 128;  // Amichevole per cache L3
    const int outer_tile_y = 32;
    const int inner_tile_x = 16;   // Amichevole per cache L1
    const int inner_tile_y = 8;
    const int vector_width = 8;    // Larghezza AVX2 per float
    
    // Tiling a tre livelli: esterno, interno, vettoriale
    mandelbrot.tile(x, y, xo, yo, xi, yi, outer_tile_x, outer_tile_y)
              .tile(xi, yi, xii, yii, inner_tile_x, inner_tile_y)
              .parallel(yo)                    // Parallelismo su tile esterni
              .vectorize(xii, vector_width)    // Vettorizzazione SIMD
              .unroll(xii, 2)                  // Parallelismo a livello istruzioni
              .unroll(yii, 2);
    
    // Scheduling delle funzioni intermedie per uso ottimale della cache
    coords.compute_at(mandelbrot, yo)
          .vectorize(x, vector_width)
          .unroll(x, 2);
    
    iteration.compute_at(mandelbrot, yi)
             .vectorize(x, vector_width)
             .unroll(x, 2);
    
    color_mapping.compute_at(mandelbrot, yi)
                 .vectorize(x, vector_width);
    
    // Storage RGB interlacciato per migliore località cache
    mandelbrot.reorder_storage(c, x, y);
    
    std::vector<Argument> args = {
        width, height, 
        xmin, xmax, ymin, ymax,
        max_iter, ncolors,
        color_lut
    };
    
    mandelbrot.compile_to_file("mandelbrot_generator", args, "mandelbrot", target);
    
    return 0;
}
