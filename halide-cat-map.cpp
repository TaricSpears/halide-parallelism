#include "Halide.h"
#include <vector>

using namespace Halide;

int main() {
    // Input: immagine quadrata N×N
    ImageParam input(UInt(8), 2, "input");
    Var x("x"), y("y");
    
    // Versione single-step: una iterazione alla volta
    Func cat_map("cat_map");
    Expr width = input.width();
    Expr height = input.height();
    
    // Trasformazione inversa: (x,y) → coordinate sorgente
    Expr x_src = (x - y + width) % width;
    Expr y_src = (-x + 2*y + height) % height;
    cat_map(x, y) = input(x_src, y_src);
    
    // Versione k-step: k iterazioni pre-calcolate
    Func cat_map_k("cat_map_k");
    int k = 100;
    
    Expr xcur = x;
    Expr ycur = y;
    for (int i = 0; i < k; i++) {
        Expr xnext = (2*xcur + ycur) % width;
        Expr ynext = (xcur + ycur) % height;
        xcur = xnext;
        ycur = ynext;
    }
    cat_map_k(x, y) = input(xcur, ycur);
    
    // Target CPU
    Target target = get_host_target();
    
    // Schedule: split semplice per parallelismo massimo
    Var yo_outer("yo_outer"), yo_inner("yo_inner");
    cat_map.split(y, yo_outer, yo_inner, 114);
    cat_map.parallel(yo_outer);
    cat_map.vectorize(x, 16);
    
    cat_map_k.split(y, yo_outer, yo_inner, 114);
    cat_map_k.parallel(yo_outer);
    cat_map_k.vectorize(x, 16);
    
    // Compilazione
    std::vector<Argument> args = {input};
    cat_map.compile_to_file("halide-catmap-single-gen", args, "cat_map_single", target);
    cat_map_k.compile_to_file("halide-catmap-k-gen", args, "cat_map_k", target);
    
    printf("Halide Cat Map generator compiled for Xeon\n");
    printf("Split scheduling: %d rows per thread, vectorization x16\n", 114);
    
    return 0;
}
