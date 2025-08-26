#include "Halide.h"
#include <vector>

using namespace Halide;

int main() {
    // Parametro di input: griglia 2D dell'automa cellulare
    ImageParam current(UInt(8), 2, "current");
    
    Var x("x"), y("y");
    
    // Implementa bordi ciclici toroidali per condizioni al contorno periodiche
    Func input_wrapped("input_wrapped");
    input_wrapped(x, y) = current((x % current.width() + current.width()) % current.width(),
                                  (y % current.height() + current.height()) % current.height());
    
    // Automa cellulare ANNEAL: regola twisted majority
    Func anneal_step("anneal_step");
    
    // Conta cellule vive nella finestra 3x3 (inclusa cella centrale)
    RDom neighbors(-1, 3, -1, 3);
    Expr nblack = sum(cast<int32_t>(input_wrapped(x + neighbors.x, y + neighbors.y)));
    
    // Regola ANNEAL: nuova_cella = 1 se (vicini >= 6 OR vicini == 4), altrimenti 0
    anneal_step(x, y) = cast<uint8_t>((nblack >= 6) || (nblack == 4));
    
    // Target ottimizzato per GTX 1070 (Pascal, CC 6.1)
    Target target = get_host_target();
    target = target.with_feature(Target::CUDA);
    target = target.with_feature(Target::CUDACapability61);
    
    // Scheduling ottimizzato per GTX 1070 (15 SM, 1920 cores)
    if (target.has_feature(Target::CUDA)) {
        Var xo("xo"), yo("yo"), xi("xi"), yi("yi");
        
        // Tiling ottimale per GTX 1070: 32x16 per massimo occupancy
        // 512 thread per block (32×16) = occupancy ottimale per SM
        anneal_step.tile(x, y, xo, yo, xi, yi, 32, 16);
        anneal_step.gpu_blocks(xo, yo);
        anneal_step.gpu_threads(xi, yi);
        
        // Inline per ridurre memory latency su Pascal
        input_wrapped.compute_inline();
        
    } else {
        // CPU fallback
        input_wrapped.compute_at(anneal_step, y);
        anneal_step.vectorize(x, 8).parallel(y);
    }
    
    // Compilazione AOT
    std::vector<Argument> args = {current};
    anneal_step.compile_to_file("halide-anneal-gen", args, "anneal_step", target);
    
    printf("Halide ANNEAL generator compiled for %s target\n", 
           target.has_feature(Target::CUDA) ? "GPU" : "CPU");
    
    return 0;
}
