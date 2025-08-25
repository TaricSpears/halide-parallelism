#include "Halide.h"
#include <vector>

using namespace Halide;

int main(int argc, char **argv) {
   // Parametri di input
   ImageParam input(UInt(8), 2, "input");     // Immagine in ingresso (grayscale)
   Param<int> threshold("threshold");          // Soglia per binarizzazione

   // Variabili di coordinata
   Var x("x"), y("y"), xo("xo"), yo("yo"), xi("xi"), yi("yi");

   // Gestione bordi: replica pixel di bordo
   Func input_clamped("input_clamped");
   input_clamped(x, y) = input(clamp(x, 0, input.width()-1), clamp(y, 0, input.height()-1));

   // Gradienti Sobel
   Func gx("gx"), gy("gy");
   
   // Gradiente orizzontale (kernel [-1 0 1; -2 0 2; -1 0 1]): rileva bordi verticali
   gx(x, y) = -cast<int32_t>(input_clamped(x-1, y-1)) + cast<int32_t>(input_clamped(x+1, y-1))
            - 2*cast<int32_t>(input_clamped(x-1, y)) + 2*cast<int32_t>(input_clamped(x+1, y))
            - cast<int32_t>(input_clamped(x-1, y+1)) + cast<int32_t>(input_clamped(x+1, y+1));

   // Gradiente verticale (kernel [-1 -2 -1; 0 0 0; 1 2 1]): rileva bordi orizzontali
   gy(x, y) = -cast<int32_t>(input_clamped(x-1, y-1)) - 2*cast<int32_t>(input_clamped(x, y-1)) - cast<int32_t>(input_clamped(x+1, y-1))
            + cast<int32_t>(input_clamped(x-1, y+1)) + 2*cast<int32_t>(input_clamped(x, y+1)) + cast<int32_t>(input_clamped(x+1, y+1));

   // Magnitudine del gradiente (Gx² + Gy²)
   Func magnitude("magnitude");
   magnitude(x, y) = gx(x, y)*gx(x, y) + gy(x, y)*gy(x, y);

   // Binarizzazione: bianco se sopra soglia, nero altrimenti
   Func sobel("sobel");
   sobel(x, y) = select(magnitude(x, y) > threshold*threshold, cast<uint8_t>(255), cast<uint8_t>(0));

   // Scheduling CPU ottimizzato
   const int tile_x = 128;
   const int tile_y = 32;

   // Tiling con parallelismo e vettorizzazione
   sobel.tile(x, y, xo, yo, xi, yi, tile_x, tile_y)
        .vectorize(xi, 8)      // Vettorizzazione SIMD
        .parallel(yo);         // Parallelismo su tile esterni

   // Compute intermediate functions a livello tile per cache locality
   gx.compute_at(sobel, yo).vectorize(x, 8);
   gy.compute_at(sobel, yo).vectorize(x, 8);
   magnitude.compute_at(sobel, yo).vectorize(x, 8);

   // Compilazione AOT
   std::vector<Argument> args = {input, threshold};
   Target target = get_host_target();

   sobel.compile_to_file("sobel_generator", args, "sobel", target);

   return 0;
}
