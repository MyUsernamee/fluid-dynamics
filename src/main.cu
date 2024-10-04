#include "raylib.h"
#include <vector>
#include <glm/glm.hpp>
#include <algorithm>
#include <string.h>
#include <functional>
#include "simulation.hpp"
#include "config.hpp"
#include <stdio.h>
#include <iostream>

#define AUDIO_BUFFER_SIZE 1024

#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

struct AudioBuffer
{

    float data[AUDIO_BUFFER_SIZE];
    int index = 0;

    __host__ __device__ void push_sample(float sample)
    {
        data[index] = sample;
        ++index;

        if (index > AUDIO_BUFFER_SIZE)
        {
            index = 0;
        }
    }
};

enum DrawMode
{

    COLOR = 0,
    PRESSURE = 1,
    VELOCITY = 2,

};

void drawArrow(glm::vec2 direction, int x, int y, int w)
{
    DrawLineEx({(float)x, (float)y}, {x + direction.x, y + direction.y}, w, WHITE);
}

template <typename T>
std::pair<T, T> min_max(std::function<T(int index)> f, int count)
{

    T min = f(0);
    T max = f(0);

    for (int i = 0; i < count; ++i)
    {
        T value = f(i);

        min = std::min(min, value);
        max = std::max(max, value);
    }

    return {min, max};
}

void draw(Grid<FluidData, N, N> data, DrawMode mode)
{
    std::pair<float, float> lim;
    float min;
    float max;
    lim = min_max<float>([&](int i)
                         { return data.data[i].getPressure(dx * dx); }, N * N);

    max = lim.second;
    min = lim.first;

    if (mode == PRESSURE)
    {
        // Draw a little box to show limits
        DrawRectangleLines(10, 9, 40, 81, WHITE);
        for (int i = 0; i < 80; ++i)
        {
            DrawLine(11, i + 10, 49, i + 10, ColorFromHSV(250.0 - std::fmod((80.0 - i) / 80.0 * 250.0, 360.0), 1.0, 1.0));
        }

        DrawText(TextFormat("%f", max), 60, 9, 12, WHITE);
        DrawText(TextFormat("%f", min), 60, 81, 12, WHITE);
    }

    for (int x = 0; x < N; ++x)
    {

        // DrawLine(start_x + x * cell_size, 0, start_x + x * cell_size, HEIGHT, WHITE);

        for (int y = 0; y < N; ++y)
        {

            Color color = BLACK;

            glm::vec3 c;
            glm::vec2 v;

            switch (mode)
            {

            case PRESSURE:

                color = ColorFromHSV(std::fmod(250.0 - (data.get(x, y).getPressure(dx * dx) - min) / (max - min) * 250.0, 360.0), 1.0, 1.0);

                // We are going to draw a handy diagram for helping visualize

                break;

            case COLOR:

                c = data.get(x, y).color;

                color = ColorFromNormalized({c.x, c.y, c.z, 1.0});

                break;

            case VELOCITY:

                v = glm::clamp(glm::abs(data.get(x, y).vel / data.get(x, y).density / 4.0f), glm::vec2(0.0), glm::vec2(1.0));
                color = ColorFromNormalized({v.x, v.y, 0.0, 1.0});

                break;
            }

            // drawArrow(data.get(x, y).vel * cell_size, start_x + x * cell_size + 0.5 * cell_size, start_y + y * cell_size + 0.5 * cell_size, 1.0);
            DrawRectangle(start_x + x * cell_size + 0.5 * cell_size, start_y + y * cell_size + 0.5 * cell_size, cell_size, cell_size,
                          color);

            if (data.get(x, y).wall)
            {
                DrawRectangle(start_x + x * cell_size + 0.5 * cell_size, start_y + y * cell_size + 0.5 * cell_size, cell_size, cell_size, RED);
            }
        }
    }
}

__global__ void update(Grid<FluidData, N, N> grid, Grid<FluidData, N, N> new_grid, glm::vec2 sample_position, float *o, int index, float dt)
{

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= N - 1 || y >= N - 1 || x < 0 || y < 0)
        return;

    if (grid.get(x, y).wall)
        return;

    applyPressureForce(grid, dt, x, y);
    advect(grid, new_grid, dt, x, y);

    if (x == sample_position.x && y == sample_position.y)
    {
        o[index] = grid.get(x, y).getPressure(dx * dx);
    }
}

int main()
{

    InitWindow(WIDTH, HEIGHT, "Fluid Dynamics");
    InitAudioDevice();

    SetAudioStreamBufferSizeDefault(AUDIO_BUFFER_SIZE);
    AudioStream stream = LoadAudioStream(5000, 16, 1);

    dim3 blocks(N / 16 + 1, N / 16 + 1);
    dim3 threads(16, 16);

    PlayAudioStream(stream);

    Grid<FluidData, N, N> grid(FluidData{glm::vec2(0.0, 0.0)});
    AudioBuffer *buffer = new AudioBuffer();

    // Make edges walls
    for (int i = 0; i < N; ++i)
    {
        grid.get(i, 0).wall = true;
        grid.get(i, N - 1).wall = true;
        grid.get(0, i).wall = true;
        grid.get(N - 1, i).wall = true;

        grid.get(i, 0).density = 0.0;
        grid.get(i, N - 1).density = 0.0;
        grid.get(0, i).density = 0.0;
        grid.get(N - 1, i).density = 0.0;
    }

    Grid<FluidData, N, N> d_grid(FluidData{glm::vec2(0.1, 0.0)});
    Grid<FluidData, N, N> d_back_grid(FluidData{glm::vec2(0.1, 0.0)});
    Grid<float, N, N> d_p_grid(0.0);
    AudioBuffer *d_buff;

    delete d_grid.data; // We don't need another array on the cpu
    delete d_back_grid.data;
    delete d_p_grid.data;

    cudaMalloc(&d_grid.data, sizeof(FluidData) * N * N);
    cudaMalloc(&d_back_grid.data, sizeof(FluidData) * N * N);
    cudaMalloc(&d_p_grid.data, sizeof(float) * N * N);
    cudaMalloc(&d_buff, sizeof(AudioBuffer));

    // SetTargetFPS(60);

    auto view_mode = PRESSURE;
    auto mouse_size = 10;
    short *data = new short[AUDIO_BUFFER_SIZE];
    int data_index = 0;
    float *d_o;
    cudaMalloc(&d_o, sizeof(float) * AUDIO_BUFFER_SIZE + 1);
    float *o = new float[AUDIO_BUFFER_SIZE + 1];

    gpuErrchk(cudaMemcpy(d_grid.data, grid.data, sizeof(FluidData) * N * N, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_back_grid.data, d_grid.data, sizeof(FluidData) * N * N, cudaMemcpyDeviceToDevice));

    while (!WindowShouldClose())
    {
        BeginDrawing();

        ClearBackground(BLACK);

        DrawFPS(10, 200);

        auto mouse_p = getLocalPosition(GetMouseX(), GetMouseY());
        auto mouse_d = GetMouseDelta();
        draw(grid, view_mode);

        // Set the pressure of the cells near the edge but not the edge edge to atmospheric pressure
        for (int i = 0; i < N - 1; ++i)
        {
            grid.get(1, i + 1).density = PVtoDensity(101325.0, 300.0);
            grid.get(N - 2, i + 1).density = PVtoDensity(101325.0, 300.0);
            grid.get(i + 1, 1).density = PVtoDensity(101325.0, 300.0);
            grid.get(i + 1, N - 2).density = PVtoDensity(101325.0, 300.0);
        }

        if (IsMouseButtonDown(MOUSE_BUTTON_LEFT))
        {
            // Make Wall
            fillCircle(grid, FluidData{glm::vec2(0.0), glm::vec3(0.0), true, 0.0}, mouse_p.x, mouse_p.y, mouse_size);
        }
        if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT))
        {
            fillCircle(grid, FluidData{(glm::vec2(mouse_d.x + 0.0001, mouse_d.y)) * 1.0f, glm::vec3(1.0, 0.0, 0.0), false, grid.get(mouse_p.x, mouse_p.y).density * 1.1f}, mouse_p.x, mouse_p.y, mouse_size);
        }
        if (IsMouseButtonDown(MOUSE_BUTTON_MIDDLE))
        {
            fillCircle(grid, FluidData{glm::vec2(0.0), glm::vec3(0.0), false, grid.get(mouse_p.x, mouse_p.y).density / 1.1f}, mouse_p.x, mouse_p.y, mouse_size);
        }

        drawArrow(getDataAtPoint(grid, mouse_p.x, mouse_p.y).vel * cell_size, GetMouseX(), GetMouseY(), 4);
        auto a = getGlobalPosition(mouse_p.x, mouse_p.y);
        DrawCircle(a.x, a.y, 4.0, RED);

        if (IsAudioStreamProcessed(stream))
        {

            cudaMemcpy(o, d_o, sizeof(float) * AUDIO_BUFFER_SIZE, cudaMemcpyDeviceToHost);

            for (int i = 1; i < AUDIO_BUFFER_SIZE; ++i)
            {
                data[i] = (short)((o[i] - o[i - 1]) * 32000. * 0.0001);
            }
            data[0] = data[1];

            UpdateAudioStream(stream, data, AUDIO_BUFFER_SIZE);
            data_index = 0;
            gpuErrchk(cudaMemcpy(d_grid.data, grid.data, sizeof(FluidData) * N * N, cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(d_back_grid.data, d_grid.data, sizeof(FluidData) * N * N, cudaMemcpyDeviceToDevice));
        }

        while (data_index < AUDIO_BUFFER_SIZE)
        {

            update<<<blocks, threads>>>(d_grid, d_back_grid, glm::vec2(10.0, 10.0), d_o, data_index, 1.0f / 5000.0f);
            gpuErrchk(cudaMemcpy(d_grid.data, d_back_grid.data, sizeof(FluidData) * N * N, cudaMemcpyDeviceToDevice));

            data_index++;
            if (data_index == AUDIO_BUFFER_SIZE - 1)
            {
                gpuErrchk(cudaMemcpy(grid.data, d_grid.data, sizeof(FluidData) * N * N, cudaMemcpyDeviceToHost));
            }
        }

        // Render rect lines to show mouse size and scale
        DrawRectangleLines(GetMouseX() - mouse_size / 2 * cell_size, GetMouseY() - mouse_size / 2 * cell_size, mouse_size * cell_size, mouse_size * cell_size, WHITE);

        // Draw text
        DrawText(TextFormat("View Mode: %d", view_mode), 10, 100, 12, WHITE);
        DrawText(TextFormat("Mouse Size: %d", mouse_size), 10, 130, 12, WHITE);

        mouse_size += GetMouseWheelMove();

        if (IsKeyPressed(KEY_SPACE))
        {
            view_mode = (DrawMode)((view_mode + 1) % 3);
        }

        EndDrawing();
    }

    CloseWindow();
}