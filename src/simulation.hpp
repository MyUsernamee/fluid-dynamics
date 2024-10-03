#include "raylib.h"
#include <vector>
#include <glm/glm.hpp>
#include <algorithm>
#include <string.h>
#include "config.hpp"

#pragma once

template <typename T>
int sgn(T val)
{ // Thank you : https://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-sgn-in-c-c
    return (T(0) < val) - (val < T(0));
}

float PVtoDensity(float p, float v)
{

    auto n = (p * v) / (R * TEMPERATURE);

    return n * MOLAR_WEIGHT_AIR / v;
}

struct FluidData
{

    glm::vec2 vel = glm::vec2(0.0);
    glm::vec3 color = glm::vec3(0.0);
    bool wall = false;
    float density = PVtoDensity(101352.9, dx *dx);

    FluidData operator+(FluidData b)
    {
        return FluidData{
            this->vel + b.vel,
            this->color + b.color,
            this->wall,
            this->density + b.density,
        };
    }

    FluidData operator*(float b)
    {
        return FluidData{
            this->vel * b,
            this->color * b,
            this->wall,
            this->density * b,
        };
    }

    FluidData operator/(float b)
    {
        return FluidData{
            this->vel / b,
            this->color / b,
            this->wall,
            this->density / b,
        };
    }

    FluidData operator-(FluidData b)
    {
        return FluidData{
            this->vel - b.vel,
            this->color - b.color,
            this->wall,
            this->density - b.density,
        };
    }

    float getPressure(float v)
    {
        return (density / MOLAR_WEIGHT_AIR) * TEMPERATURE * R * 10000000.0;
    }
};

float fract(float x)
{
    return x - floor(x);
}

template <typename T, size_t I, size_t J>
struct Grid
{
    T *data;

    Grid(T default_value)
    {

        data = new T[I * J];

        for (int i = 0; i < I * J; ++i)
        {
            data[i] = default_value;
        }
    }

    T &get(int x, int y)
    {
        return data[x + y * I];
    }

    void set(int x, int y, T t)
    {
        data[x + y * I] = t;
    }
};

bool inBounds(float x, float y)
{
    return x >= 0 && x < N && y >= 0 && y < N;
}

FluidData getDataAtPoint(Grid<FluidData, N, N> data, float x, float y)
{

    FluidData vel{

    };

    vel.density = 0.0;

    for (int i = 0; i <= 1; ++i)
    {
        for (int j = 0; j <= 1; ++j)
        {
            if (!inBounds(x + i, y + j))
                continue;
            vel = vel + data.get(x + i, y + j) * (1.0f - std::abs(fract(x) - i)) * (1.0f - std::abs(fract(y) - j));
        }
    }
    return vel;
}

Grid<float, N, N> calculateDivergence(Grid<FluidData, N, N> &data)
{

    auto out = Grid<float, N, N>(0.0);

#pragma omp parallel for
    for (int x = 0; x < N; ++x)
    {

#pragma omp simd
        for (int y = 0; y < N; ++y)
        {
            float left_edge_flow = (x > 0 ? (data.get(x, y).vel.x + data.get(x - 1, y).vel.x) * 0.5f : data.get(x, y).vel.x);
            float right_edge_flow = (x < N - 1 ? (data.get(x, y).vel.x + data.get(x + 1, y).vel.x) * 0.5f : data.get(x, y).vel.x);
            float bottom_edge_flow = (y > 0 ? (data.get(x, y).vel.y + data.get(x, y - 1).vel.y) * 0.5f : data.get(x, y).vel.y);
            float top_edge_flow = (y < N - 1 ? (data.get(x, y).vel.y + data.get(x, y + 1).vel.y) * 0.5f : data.get(x, y).vel.y);

            out.set(x, y, (right_edge_flow - left_edge_flow + top_edge_flow - bottom_edge_flow));
        }
    }

    return out;
}

void applyPressureForce(Grid<FluidData, N, N> &data, float dt)
{

    for (int x = 1; x < N - 1; ++x)
    {
        for (int y = 1; y < N - 1; ++y)

        {

            for (int i = -1; i <= 1; ++i)
            {
                for (int j = -1; j <= 1; ++j)
                {
                    if (abs(i) == abs(j))
                        continue; // Exclude conners and self
                    glm::vec2 offset(i, j);

                    // Calculate gradient of pressure
                    auto grad = -(data.get(x, y).getPressure(dx * dx) - data.get(x + i, y + j).getPressure(dx * dx)) * offset * dx;

                    // Apply force
                    data.get(x, y).vel += (-grad / data.get(x, y).density) * dt;
                }
            }
        }
    }
}

void advect(Grid<FluidData, N, N> &data, float dt)
{

    auto new_grid = Grid<FluidData, N, N>(FluidData{});

#pragma omp parallel for
    for (int x = 1; x < N - 1; ++x)
    {
#pragma omp parallel for
        for (int y = 1; y < N - 1; ++y)
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
                    auto other = data.get(x + i, y + j);

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
    }

    delete data.data;
    data = new_grid;
}

void fillCircle(Grid<FluidData, N, N> &data, FluidData new_data, int x, int y, float r)
{

    for (int i = -r / 2.0; i < r / 2.0; ++i)
    {

        for (int j = -r / 2.0; j < r / 2.0; ++j)
        {
            data.set(x + i, y + j, new_data);
        }
    }
}

void integrate(Grid<FluidData, N, N> &data, float dt)
{

    for (int x = 0; x < N; ++x)
    {
        for (int y = 0; y < N; ++y)
        {

            data.get(x, y).vel -= dt * glm::vec2(0.0, 1.0);
        }
    }
}

Vector2 getLocalPosition(int x, int y)
{
    return {(x - start_x) / cell_size, (y - start_y) / cell_size};
}

Vector2 getGlobalPosition(int x, int y)
{

    return {x * cell_size + start_x + cell_size / 2.0f, y * cell_size + start_y + cell_size / 2.0f};
}