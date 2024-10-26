#include "raylib.h"
#include <vector>
#include <glm/glm.hpp>
#include <algorithm>
#include <string.h>
#include <stdio.h>
#include "consts.hpp"
#pragma once

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

template <typename T>
int sgn(T val)
{ // Thank you : https://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-sgn-in-c-c
    return (T(0) < val) - (val < T(0));
}

__device__ __host__ float PVtoDensity(float p, float v)
{

    auto n = (p * v) / (R * TEMPERATURE);

    return n * MOLAR_WEIGHT_AIR / v;
}

__host__ __device__ float DensityToPV(float d, float v)
{

    auto n = d * v / MOLAR_WEIGHT_AIR;
    return n * R * TEMPERATURE / v * 1000000.0;
}

struct FluidData
{

    glm::vec2 vel = glm::vec2(0.0);
    glm::vec3 color = glm::vec3(0.0);
    bool wall = false;
    float dx;
    float density = PVtoDensity(101352.9, dx *dx);
    float energy = 0.0;

    FluidData(float pressure, float temperature, float _dx) : dx(_dx)
    {
        density = PVtoDensity(pressure, dx * dx);
        energy = (5.0 / 2.0) * pressure * dx * dx;
    }

    __device__ __host__ FluidData() {}

    __device__ __host__ FluidData(glm::vec2 vel, glm::vec3 color, bool wall, float density, float energy, float dx)
    {
        this->vel = vel;
        this->color = color;
        this->wall = wall;
        this->density = density;
        this->energy = energy;
        this->dx = dx;
    }

    FluidData(glm::vec2 vel, glm::vec3 color, bool wall, float pressure, float dx)
    {
        this->vel = vel;
        this->color = color;
        this->wall = wall;
        this->density = PVtoDensity(pressure, dx * dx);
        this->energy = (5.0 / 2.0) * pressure * dx * dx;
        this->dx = dx;
    }

    __device__ __host__ FluidData operator+(FluidData b)
    {
        return FluidData{
            this->vel + b.vel,
            this->color + b.color,
            this->wall,
            this->density + b.density,
            this->energy + b.energy,
            this->dx + b.dx};
    }

    __device__ __host__ FluidData operator*(float b)
    {
        return FluidData{
            this->vel * b,
            this->color * b,
            this->wall,
            this->density * b,
            this->energy * b,
            this->dx * b};
    }

    __device__ __host__ FluidData operator/(float b)
    {
        return FluidData{
            this->vel / b,
            this->color / b,
            this->wall,
            this->density / b,
            this->energy / b,
            this->dx / b};
    }

    __device__ __host__ FluidData operator-(FluidData b)
    {
        return FluidData{
            this->vel - b.vel,
            this->color - b.color,
            this->wall,
            this->density - b.density,
            this->energy - b.energy,
            this->dx - b.dx};
    }

    __device__ __host__ float getPressure(float v)
    {

        if (wall)
            return 0;
        float pressure = (2.0 / 5.0) * energy / v;
        return pressure;
    }

    __device__ __host__ void setPressure(float pressure)
    {
        energy = (5.0 / 2.0) * pressure * dx * dx;
        density = PVtoDensity(pressure, dx * dx);
    }
};

float fract(float x)
{
    return x - floor(x);
}

template <typename T>
struct Grid
{
    T *data;

    int width;
    int height;

    Grid(T default_value, int width, int height)
    {

        this->width = width;
        this->height = height;

        data = new T[width * height];

        for (int i = 0; i < width * height; ++i)
        {
            data[i] = default_value;
        }
    }

    Grid(int _width, int _height) : width(_width), height(_height)
    {
    }

    Grid()
    {
    }

    __device__ __host__ T &get(int x, int y)
    {
        return data[x + y * width];
    }

    __device__ __host__ void set(int x, int y, T t)
    {
        data[x + y * width] = t;
    }
};
enum BoundaryCondition
{
    CLOSED,
    REPEAT,
    OPEN,
    ATM
};

__host__ __device__ bool inBounds(float x, float y, int width, int height)
{
    return x >= 0 && x < (float)width && y >= 0 && y < (float)height;
}

FluidData getDataAtPoint(Grid<FluidData> data, float x, float y)
{

    FluidData vel{};

    vel.density = 0.0;

    for (int i = 0; i <= 1; ++i)
    {
        for (int j = 0; j <= 1; ++j)
        {
            if (!inBounds(x + i, y + j, data.width, data.height))
                continue;
            vel = vel + data.get(x + i, y + j) * (1.0f - std::abs(fract(x) - i)) * (1.0f - std::abs(fract(y) - j));
        }
    }
    return vel;
}

__device__ void applyPressureForce(Grid<FluidData> data, float dt, int x, int y, float dx)
{

    for (int i = -1; i <= 1; ++i)
    {
        for (int j = -1; j <= 1; ++j)
        {
            if (abs(i) == abs(j))
                continue; // Exclude conners and self
            glm::vec2 offset(i, j);

            if (data.get(x + i, y + j).wall || !inBounds(x + i, y + j, data.width, data.height))
            {

                continue;
            }

            // Calculate gradient of pressure
            auto grad = -(data.get(x, y).getPressure(dx * dx) - data.get(x + i, y + j).getPressure(dx * dx)) * offset * dx;

            // Apply force
            data.get(x, y).vel += (-grad) * dt;
        }
    }
}

__device__ void advect(Grid<FluidData> &data, Grid<FluidData> &new_grid, BoundaryCondition bc, float dt, int x, int y, float dx)
{

    auto self = data.get(x, y);
    new_grid.set(x, y, self);
    auto &new_self = new_grid.get(x, y);
    for (int i = -1; i <= 1; ++i)
    {

        for (int j = -1; j <= 1; ++j)
        {

            if (abs(i) == abs(j))
                continue; // Ignore corners and self
            glm::vec2 offset(i, j);
            FluidData other;
            if (inBounds(x + i, y + j, data.width, data.height))
                other = data.get(x + i, y + j);
            else
            {
                // Cancel out velocity going in this direction
                // self.vel -= self.vel * max(glm::dot(self.vel, offset), 0.0f);
                continue;
            }

            if (other.wall)
            {
                // self.vel -= self.vel * max(glm::dot(self.vel, offset), 0.0f);
                continue;
            }
            // First we calculate the velocity at the boundary
            auto v = (self.vel / self.density + other.vel / other.density) * 0.5f;
            // Fix it to the direction we are workingin
            v *= glm::abs(offset);
            auto v_direction = glm::dot(glm::vec2(1.0), v); // Same as 1.0 * x + 1.0 * y;
            auto direction = glm::dot(offset, glm::vec2(1.0));
            // And one of x or y should be 0.

            if (v_direction != 0.0)
                // TraceLog(LOG_WARNING, TextFormat("%f", v_direction));

                // If the velocity is pointing into our cell
                if (v_direction * direction < 0)
                {

                    // We then advect it in
                    new_self = new_self + (other * std::abs(v_direction)) * (dt / dx);
                }
                else
                {
                    // If it is pointing out of our cell, we should advect it out
                    new_self = new_self - (self * std::abs(v_direction)) * (dt / dx);
                }
        }
    }
}

void fillCircle(Grid<FluidData> &data, FluidData new_data, int x, int y, float r)
{

    for (int i = -r / 2.0; i < r / 2.0; ++i)
    {

        for (int j = -r / 2.0; j < r / 2.0; ++j)
        {
            data.set(x + i, y + j, new_data);
        }
    }
}

void setCircleColor(Grid<FluidData> &data, glm::vec3 color, int x, int y, float r)
{

    for (int i = -r / 2.0; i < r / 2.0; ++i)
    {

        for (int j = -r / 2.0; j < r / 2.0; ++j)
        {
            data.get(x + i, y + j).color = color;
        }
    }
}

__device__ void integrate(Grid<FluidData> &data, float dt, int x, int y)
{

    data.get(x, y).vel -= dt * glm::vec2(0.0, 9.8);
}

__global__ void update(Grid<FluidData> grid, Grid<FluidData> new_grid, int width, int height, float dt, float dx)
{

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= width || y >= height || x < 0 || y < 0)
        return;

    if (grid.get(x, y).wall)
        return;

    applyPressureForce(grid, dt, x, y, dx);
    advect(grid, new_grid, BoundaryCondition::CLOSED, dt, x, y, dx);
    // integrate(new_grid, dt, x, y);
    grid.set(x, y, new_grid.get(x, y));
}

class Simulation
{

private:
    Grid<FluidData> d_data;
    Grid<FluidData> d_back_data;
    dim3 threads = dim3(16, 16);
    dim3 blocks;

public:
    int N;
    Grid<FluidData> data;
    FluidData defualt_value;
    float dx;

    BoundaryCondition bc;

    void init()
    {
        d_data = {data.width, data.height};
        d_back_data = {data.width, data.height};

        gpuErrchk(cudaMalloc(&d_data.data, sizeof(FluidData) * N));
        gpuErrchk(cudaMalloc(&d_back_data.data, sizeof(FluidData) * N));
    }

    Simulation(float size, int width, int height, BoundaryCondition condition, FluidData dv)
    {
        data = Grid<FluidData>(dv, width, height);
        this->defualt_value = dv;
        blocks = dim3(width / threads.x + 1, height / threads.y + 1);
        N = width * height;
        dx = size / (float)width;
        bc = condition;
        init();
    }

    void setValue(FluidData value, int x, int y)
    {
        data.set(x, y, value);
    }

    FluidData &getValue(int x, int y)
    {
        return data.get(x, y);
    }

    void step(float dt)
    {
        syncDataToGPU();
        stepNoSync(dt);
        syncDataToCPU();
    }

    void syncDataToCPU()
    {
        gpuErrchk(cudaMemcpy(data.data, d_data.data, sizeof(FluidData) * N, cudaMemcpyDeviceToHost));
    }

    void syncDataToGPU()
    {
        gpuErrchk(cudaMemcpy(d_data.data, data.data, sizeof(FluidData) * N, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_back_data.data, d_data.data, sizeof(FluidData) * N, cudaMemcpyDeviceToDevice));
    }

    void stepNoSync(float dt)
    {
        update<<<blocks, threads>>>(d_data, d_back_data, data.width, data.height, dt, dx);
        // cudaMemcpy(d_data.data, d_back_data.data, sizeof(FluidData) * N, cudaMemcpyDeviceToDevice);
    }

    void applyBoundaryConditions()
    {

        switch (bc)
        {
        }
    }

    float getSafeDT(float dt)
    {

        for (int i = 0; i < N; i++)
        {

            auto cell = data.data[i];
            if (cell.wall)
                continue;
            auto new_dt = dx / (glm::length(cell.vel) / cell.density) * 0.5f;
            dt = min(dt, new_dt);
        }

        return dt;
    }
};