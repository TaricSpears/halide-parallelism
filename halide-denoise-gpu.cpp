#include "Halide.h"
#include <vector>

using namespace Halide;

int main(int argc, char **argv) {
    ImageParam input(UInt(8), 2, "input");
    
    Var x("x"), y("y");
    Var xo("xo"), yo("yo"), xi("xi"), yi("yi");
    Var xii("xii"), yii("yii");
    
    Func denoised("denoised");
    
    // Algoritmo di mediana: campiona 5 pixel (centro + 4 adiacenti)
    Expr v0 = cast<int32_t>(input(clamp(x, 0, input.width()-1), clamp(y, 0, input.height()-1)));
    Expr v1 = cast<int32_t>(input(clamp(x-1, 0, input.width()-1), clamp(y, 0, input.height()-1)));
    Expr v2 = cast<int32_t>(input(clamp(x+1, 0, input.width()-1), clamp(y, 0, input.height()-1)));
    Expr v3 = cast<int32_t>(input(clamp(x, 0, input.width()-1), clamp(y-1, 0, input.height()-1)));
    Expr v4 = cast<int32_t>(input(clamp(x, 0, input.width()-1), clamp(y+1, 0, input.height()-1)));
    
    // Sorting network per trovare la mediana dei 5 valori
    Expr s1_min34 = min(v3, v4);
    Expr s1_max34 = max(v3, v4);
    Expr s1_min23 = min(v2, s1_min34);  
    Expr s1_max23 = max(v2, s1_min34);
    Expr s1_min12 = min(v1, s1_min23);
    Expr s1_max12 = max(v1, s1_min23);
    Expr s1_min01 = min(v0, s1_min12);
    Expr s1_max01 = max(v0, s1_min12);
    
    Expr s2_min34 = min(s1_max23, s1_max34);
    Expr s2_max34 = max(s1_max23, s1_max34);
    Expr s2_min23 = min(s1_max12, s2_min34);
    Expr s2_max23 = max(s1_max12, s2_min34);
    Expr s2_min12 = min(s1_max01, s2_min23);
    Expr s2_max12 = max(s1_max01, s2_min23);

    denoised(x, y) = cast<uint8_t>(clamp(s2_max12, 0, 255));
    
    // Scheduling ottimizzato per GPU
    const int BLOCK_SIZE_X = 32;  // Thread block dimensions
    const int BLOCK_SIZE_Y = 16;  // 32x16 = 512 threads per block
    const int TILE_SIZE_X = 64;   // Tile size per cache locality  
    const int TILE_SIZE_Y = 32;
    
    // Tiling gerarchico a due livelli
    denoised.tile(x, y, xo, yo, xi, yi, TILE_SIZE_X, TILE_SIZE_Y);
    denoised.tile(xi, yi, xii, yii, BLOCK_SIZE_X, BLOCK_SIZE_Y);
    
    // Mapping GPU: blocks e threads
    denoised.gpu_blocks(xo, yo);
    denoised.gpu_threads(xii, yii);
    
    // Cache input in shared memory per ridurre accessi global memory
    Func input_cached("input_cached");
    input_cached(x, y) = input(clamp(x, 0, input.width()-1), clamp(y, 0, input.height()-1));
    input_cached.compute_at(denoised, xo);
    input_cached.store_in(MemoryType::GPUShared);
    
    // Unroll per ridurre overhead
    denoised.unroll(xi, 2);
    
    // Target CUDA con compute capability 6.1
    Target target = get_host_target();
    target = target.with_feature(Target::CUDA);
    target = target.with_feature(Target::CUDACapability61);
    
    std::vector<Argument> args = {input};
    denoised.compile_to_file("halide-denoise-gpu-gen", args, "halide_denoise_gpu", target);
    
    return 0;
}
