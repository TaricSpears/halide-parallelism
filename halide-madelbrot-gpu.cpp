#include "Halide.h"

using namespace Halide;

int main() {
   // Parametri di input per la generazione dell'immagine
   Param<int> width, height;                    // Dimensioni immagine
   Param<float> xmin, xmax, ymin, ymax;        // Limiti del piano complesso
   Param<int> max_iter, ncolors;               // Iterazioni max e numero colori
   ImageParam color_lut(UInt(8), 2);           // Look-up table per i colori

   // Variabili di coordinata
   Var x, y, c;
   Func f;

   // Mappatura da coordinate pixel a coordinate nel piano complesso
   Expr re = xmin + (xmax - xmin) * x / (width - 1);   // Parte reale
   Expr im = ymax - (ymax - ymin) * y / (height - 1);  // Parte immaginaria

   // Inizializzazione sequenza z_n
   Expr zr = 0.0f, zi = 0.0f, iter = 0;

   // Iterazione della sequenza z_{n+1} = z_n^2 + c (16 iterazioni)
   for(int i = 0; i < 16; i++) {
       Expr zr2 = zr * zr, zi2 = zi * zi;              // Calcolo z_n^2
       Expr bounded = (zr2 + zi2) <= 4.0f;            // Test convergenza |z_n|^2 <= 4
       
       // z_{n+1} = z_n^2 + c se convergente, altrimenti mantieni z_n
       zr = select(bounded, zr2 - zi2 + re, zr);       // Parte reale
       zi = select(bounded, 2*zr*zi + im, zi);         // Parte immaginaria
       iter = select(bounded, iter + 1, iter);         // Conta iterazioni
   }

   // Mappatura colori: converte iterazioni in colore RGB
   Expr color_idx = cast<int>(iter) % ncolors;
   f(x, y, c) = cast<uint8_t>(color_lut(c, color_idx));

   // Target GPU con scheduling minimale
   Target target = get_host_target().with_feature(Target::CUDA);

   // Tiling GPU semplice con blocchi 8x8
   Var xi, yi;
   f.gpu_tile(x, y, xi, yi, 8, 8);

   // Compilazione
   std::vector<Argument> args = {width, height, xmin, xmax, ymin, ymax, max_iter, ncolors, color_lut};
   f.compile_to_file("mandelbrot_generator", args, "mandelbrot", target);
   
   return 0;
}
