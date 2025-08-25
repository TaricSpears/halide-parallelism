#include "Halide.h"
#include <vector>

using namespace Halide;

// Calcola la mediana di 5 elementi usando sorting network
Expr median_of_five_cpu(Expr v0, Expr v1, Expr v2, Expr v3, Expr v4) {
    // Converte in interi per evitare overflow
    v0 = cast<int32_t>(v0);
    v1 = cast<int32_t>(v1);
    v2 = cast<int32_t>(v2);
    v3 = cast<int32_t>(v3);
    v4 = cast<int32_t>(v4);
    
    // Sorting network: sequenza di compare_and_swap
    // Prima fase
    Expr min34 = min(v3, v4), max34 = max(v3, v4);
    v3 = min34; v4 = max34;
    
    Expr min23 = min(v2, v3), max23 = max(v2, v3);
    v2 = min23; v3 = max23;
    
    Expr min12 = min(v1, v2), max12 = max(v1, v2);
    v1 = min12; v2 = max12;
    
    Expr min01 = min(v0, v1), max01 = max(v0, v1);
    v0 = min01; v1 = max01;
    
    // Seconda fase
    min34 = min(v3, v4); max34 = max(v3, v4);
    v3 = min34; v4 = max34;
    
    min23 = min(v2, v3); max23 = max(v2, v3);
    v2 = min23; v3 = max23;
    
    min12 = min(v1, v2); max12 = max(v1, v2);
    v1 = min12; v2 = max12;
    
    // Terza fase
    min34 = min(v3, v4); max34 = max(v3, v4);
    v3 = min34; v4 = max34;
    
    min23 = min(v2, v3); max23 = max(v2, v3);
    v2 = min23; v3 = max23;
    
    // La mediana è v2 (elemento centrale)
    return cast<uint8_t>(clamp(v2, 0, 255));
}

int main(int argc, char **argv) {
    // Parametro di input: immagine 2D a 8 bit
    ImageParam input(UInt(8), 2, "input");
    
    // Variabili di coordinata e tiling
    Var x("x"), y("y");
    Var yo("yo"), yi("yi"), xo("xo"), xi("xi");
    
    Func denoised("denoised");
    
    // Applica filtro mediana su 5 pixel (centro + 4 adiacenti)
    // Usa clamp per gestire i bordi dell'immagine
    denoised(x, y) = median_of_five_cpu(
        input(clamp(x, 0, input.width()-1), clamp(y, 0, input.height()-1)),     // centro
        input(clamp(x-1, 0, input.width()-1), clamp(y, 0, input.height()-1)),   // sinistra
        input(clamp(x+1, 0, input.width()-1), clamp(y, 0, input.height()-1)),   // destra
        input(clamp(x, 0, input.width()-1), clamp(y-1, 0, input.height()-1)),   // sopra
        input(clamp(x, 0, input.width()-1), clamp(y+1, 0, input.height()-1))    // sotto
    );
    
    // Scheduling CPU ottimizzato
    const int VECTOR_SIZE = 8;      // Vettorizzazione SIMD
    const int TILE_SIZE_X = 64;     // Tile X per cache locality
    const int TILE_SIZE_Y = 32;     // Tile Y
    
    // Tiling per migliorare cache locality
    denoised.tile(x, y, xo, yo, xi, yi, TILE_SIZE_X, TILE_SIZE_Y);
    
    // Parallelizzazione nativa di Halide (thread nativi, non OpenMP)
    denoised.parallel(yo);
    
    // Vettorizzazione SIMD del loop interno
    denoised.vectorize(xi, VECTOR_SIZE);
    
    // Unroll per ridurre overhead
    denoised.unroll(yi, 2);
    
    // Target CPU con ottimizzazioni SIMD
    Target target = get_host_target();
    
    if (target.arch == Target::X86) {
        target = target.with_feature(Target::SSE41)
                       .with_feature(Target::AVX)
                       .with_feature(Target::AVX2);
    }
    
    // Compilazione
    std::vector<Argument> args = {input};
    denoised.compile_to_file("halide-denoise-cpu-gen", args, "halide_denoise_cpu", target);
    
    return 0;
}
