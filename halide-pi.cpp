#include "Halide.h"
#include <vector>

using namespace Halide;

int main() {
    const int MAX_POINTS = 10000000;
    Var i("i");
    
    // Genera risultati binari: 1 se punto dentro cerchio, 0 altrimenti
    Func monte_carlo_points("monte_carlo_points");
    
    // RNG thread-safe: seed diverso per chunk di 1000 punti
    Expr base_seed = cast<uint32_t>(17 + 19 * (i / 1000));
    Expr seed1 = base_seed + cast<uint32_t>(i * 1664525 + 1013904223);
    Expr seed2 = seed1 * 1103515245 + 12345;
    
    // Coordinate casuali in [-1, 1]
    Expr x = (2.0f * cast<float>(seed1 & 0x7FFFFFFF) / 2147483647.0f) - 1.0f;
    Expr y = (2.0f * cast<float>(seed2 & 0x7FFFFFFF) / 2147483647.0f) - 1.0f;
    
    // Test distanza euclidea: dentro cerchio unitario se x² + y² <= 1
    monte_carlo_points(i) = cast<int32_t>(x*x + y*y <= 1.0f);
    
    // Schedule per CPU multicore
    Target target = get_host_target();
    monte_carlo_points.parallel(i);
    monte_carlo_points.vectorize(i, 16);
    
    // Compilazione
    std::vector<Argument> args;
    monte_carlo_points.compile_to_file("halide-montecarlo-gen", args, "monte_carlo_points", target);
    
    printf("Monte Carlo generator compiled (max %d points)\n", MAX_POINTS);
    
    return 0;
}
