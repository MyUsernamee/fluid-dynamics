#include "raylib.h"
#include <vector>
#include <glm/glm.hpp>
#include <algorithm>
#include <string.h>
#include <functional>
#include "simulation.hpp"
#include "config.hpp"

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
    }

    for (int x = 0; x < N; ++x)
    {

        // DrawLine(start_x + x * cell_size, 0, start_x + x * cell_size, HEIGHT, WHITE);

        for (int y = 0; y < N; ++y)
        {

            Color color = BLACK;

            glm::vec3 c;

            switch (mode)
            {

            case PRESSURE:

                color = ColorFromHSV(std::fmod((data.get(x, y).getPressure(dx * dx) - min) / (max - min) * 250.0, 360.0), 1.0, 1.0);

                // We are going to draw a handy diagram for helping visualize

                break;

            case COLOR:

                c = data.get(x, y).color;

                color = ColorFromNormalized({c.x, c.y, c.z, 1.0});

                break;

            case VELOCITY:

                color = ColorFromNormalized({data.get(x, y).vel.x, data.get(x, y).vel.y, 0.0, 1.0});

                break;
            }

            // drawArrow(data.get(x, y).vel * cell_size, start_x + x * cell_size + 0.5 * cell_size, start_y + y * cell_size + 0.5 * cell_size, 1.0);
            DrawRectangle(start_x + x * cell_size + 0.5 * cell_size, start_y + y * cell_size + 0.5 * cell_size, cell_size, cell_size,
                          color);

            if (data.get(x, y).wall)
            {
                DrawRectangle(start_x + x * cell_size, start_y + y * cell_size, cell_size, cell_size, RED);
            }
        }
    }
}

int main()
{

    InitWindow(WIDTH, HEIGHT, "Fluid Dynamics");

    Grid<FluidData, N, N> grid(FluidData{glm::vec2(0.1, 0.0)});

    SetTargetFPS(60);

    while (!WindowShouldClose())
    {
        BeginDrawing();

        ClearBackground(BLACK);

        draw(grid, COLOR);

        auto mouse_p = getLocalPosition(GetMouseX(), GetMouseY());
        auto mouse_d = GetMouseDelta();

        drawArrow(getDataAtPoint(grid, mouse_p.x, mouse_p.y).vel * cell_size, GetMouseX(), GetMouseY(), 4);

        for (int i = 0; i < 100; ++i)
        {
            advect(grid, 1.0 / 60.0 / 100.0);
            applyPressureForce(grid, 1.0 / 60.0 / 100.0);
        }

        integrate(grid, 1.0 / 60.0);

        auto a = getGlobalPosition(mouse_p.x, mouse_p.y);
        DrawCircle(a.x, a.y, 4.0, RED);

        if (IsMouseButtonDown(MOUSE_BUTTON_LEFT))
        {
            fillCircle(grid, grid.get(mouse_p.x, mouse_p.y), mouse_p.x, mouse_p.y, 20);
        }
        if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT))
        {
            fillCircle(grid, FluidData{(glm::vec2(mouse_d.x + 0.0001, mouse_d.y)) * 1.0f, glm::vec3(1.0, 0.0, 0.0), false, PVtoDensity(211352.9, dx * dx)}, mouse_p.x, mouse_p.y, 20);
        }

        EndDrawing();
    }

    CloseWindow();
}