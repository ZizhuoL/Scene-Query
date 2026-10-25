// CPU reference alpha compositor for validation of trained Gaussian scenes.
// Projection conventions are checked against Apache-2.0 Brush v0.3.0;
// see ATTRIBUTION.md. This is not a training renderer.
#include <algorithm>
#include <cmath>
#include <vector>

extern "C" void blend_gaussians(const float* rows, int count, int width, int height,
                                float* output) {
    const int pixels = width * height;
    std::vector<float> transmission(pixels, 1.f);
    std::vector<unsigned char> done(pixels, 0);
    std::fill(output, output + pixels * 5, 0.f);
    for (int i = 0; i < count; ++i) {
        const float* g = rows + i * 12;
        int x0 = std::max(0, static_cast<int>(std::floor(g[0] - g[10])));
        int x1 = std::min(width - 1, static_cast<int>(std::ceil(g[0] + g[10])));
        int y0 = std::max(0, static_cast<int>(std::floor(g[1] - g[11])));
        int y1 = std::min(height - 1, static_cast<int>(std::ceil(g[1] + g[11])));
        for (int y = y0; y <= y1; ++y) {
            for (int x = x0; x <= x1; ++x) {
                const int p = y * width + x;
                if (done[p]) continue;
                const float dx = g[0] - (x + .5f), dy = g[1] - (y + .5f);
                const float exponent = .5f * (g[2]*dx*dx + g[4]*dy*dy) + g[3]*dx*dy;
                const float alpha = std::min(.999f, g[8] * std::exp(-exponent));
                if (exponent < 0.f || alpha < 1.f/255.f) continue;
                const float next = transmission[p] * (1.f-alpha);
                // Match native early termination, including omission of the last term.
                if (next <= 1e-4f) { done[p] = 1; continue; }
                const float weight = alpha * transmission[p];
                for (int c = 0; c < 3; ++c)
                    output[p*5+c] += weight * std::max(0.f, g[5+c]);
                output[p*5+3] += weight * g[9];
                transmission[p] = next;
            }
        }
    }
    for (int p = 0; p < pixels; ++p) {
        output[p*5+3] += transmission[p] * 3.5f;
        output[p*5+4] = 1.f - transmission[p];
    }
}
